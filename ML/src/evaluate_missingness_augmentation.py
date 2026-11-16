"""Compare frozen A/B augmentation to P9 and P8 under the same evaluation settings."""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from evaluate_eicu_demo import BOOTSTRAPS, MODES, SEED, evaluate_predictions, ranking, score_frozen
from prepare import OUT, sha
from stress_missingness_c import (
    C_FEATURES, C_LABELS, C_PREDICTIONS, COVERAGE, FROZEN, METRICS, MODELS,
    P7_PROTOCOL, THRESHOLD, load_cohort, mask_plan, score_metrics, stress,
)
from stress_missingness_c import REPEATS as C_REPEATS
from stress_missingness_c import SEED as C_SEED
from train_missingness_augmentation import DEFAULT_OUTPUT as DEFAULT_MODELS


EICU = OUT / "eicu_demo_external"
P9 = OUT / "missingness_stress_C"
DEFAULT_OUTPUT = OUT / "missingness_augmentation_evaluation"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def augmented_models(directory):
    paths = {model: directory / f"{model}.joblib" for model in MODELS}
    manifest = read_json(directory / "manifest.json")
    original = read_json(P7_PROTOCOL)["model_sha256"]
    if manifest["training_cohort"] != "P12 A/B" or manifest["new_fits"] != 11:
        raise ValueError("Expected completed A/B-only augmentation training")
    if manifest["original_model_sha256"] != original or {model: sha(path) for model, path in paths.items()} != manifest["model_sha256"]:
        raise ValueError("The frozen model manifests do not match")
    meta = read_json(FROZEN / "metadata.json")
    pipes = {}
    for model, path in paths.items():
        pipe, info, columns = joblib.load(path)
        estimator = pipe.named_steps["model"]
        if not isinstance(estimator, XGBClassifier) or pipe.classes_.tolist() != [0, 1]:
            raise ValueError(f"Unexpected augmented estimator for {model}")
        if columns != meta[model]["columns"] or info["fold"] != "all_AB" or info["recipe"] != manifest["selected_recipe"]:
            raise ValueError(f"Unexpected augmented schema or recipe for {model}")
        estimator.set_params(device="cpu", n_jobs=2)
        if estimator.get_xgb_params().get("device") != "cpu":
            raise RuntimeError("CPU selection failed")
        pipes[model] = (pipe, columns)
    return paths, pipes, manifest


def compare_set_c(pipes, originals):
    saved = read_json(P9 / "summary.json")
    if saved["model_sha256"] != originals:
        raise ValueError("P9 did not use the same original frozen models")
    if (saved["seed"], saved["matched_repeats"], saved["threshold"]) != (C_SEED, C_REPEATS, THRESHOLD):
        raise ValueError("P9 masking settings differ")
    for path in (C_FEATURES, C_LABELS, C_PREDICTIONS, COVERAGE):
        if sha(path) != saved["source_sha256"][path.name]:
            raise ValueError(f"P9 source changed: {path.name}")

    features, y, _ = load_cohort()
    positions, plan = mask_plan(features)
    if plan != saved["mask_plan"]:
        raise ValueError("Stress-test target counts differ from P9")
    reference = {
        f"score_{name}": pipe.predict_proba(features.loc[:, columns])[:, 1]
        for name, (pipe, columns) in pipes.items()
    }
    print("Evaluating frozen augmentation on the original Set C and P9 masks", flush=True)
    baseline, new = stress(features, y, pd.DataFrame(reference, index=features.index), pipes, positions, plan)
    old = pd.read_csv(P9 / "replicate_metrics.csv")
    keys = ["scenario", "model", "repeat", "design", "variables"]
    paired = old[keys + list(METRICS)].merge(new[keys + list(METRICS)], on=keys, suffixes=("_frozen", "_augmented"), validate="one_to_one")
    if not len(old) == len(new) == len(paired) == 336:
        raise ValueError("P9 mask replicates differ")
    output = []
    for model in MODELS:
        result = {"scenario": "clean", "model": model, "repeats": 1}
        for metric in METRICS:
            old_value, new_value = saved["baseline"][model][metric], baseline[model][metric]
            result.update({f"{metric}_frozen": old_value, f"{metric}_augmented": new_value, f"delta_{metric}": new_value - old_value})
        output.append(result)
    for (scenario, model), rows in paired.groupby(["scenario", "model"], sort=False):
        result = {"scenario": scenario, "model": model, "repeats": len(rows)}
        for metric in METRICS:
            old_col, new_col = f"{metric}_frozen", f"{metric}_augmented"
            changes = rows[new_col] - rows[old_col]
            result.update({
                old_col: float(rows[old_col].mean()), new_col: float(rows[new_col].mean()),
                f"delta_{metric}": float(changes.mean()),
                f"delta_{metric}_p10_p90": changes.quantile([0.1, 0.9]).tolist(),
            })
        output.append(result)
    return output, paired


def paired_intervals(cohort, frozen, augmented):
    y = cohort.y.to_numpy()
    groups = list(cohort.groupby("uniquepid", sort=False).indices.values())
    rng = np.random.default_rng(SEED)
    differences = {key: [] for key in frozen}
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(len(groups), size=len(groups))
        positions = np.concatenate([groups[index] for index in sampled])
        truth = y[positions]
        if truth.min() == truth.max():
            continue
        for key in differences:
            old = ranking(truth, frozen[key][positions])
            new = ranking(truth, augmented[key][positions])
            differences[key].append(np.subtract(new, old))

    intervals = {}
    for key, samples in differences.items():
        if len(samples) < BOOTSTRAPS * 0.95:
            raise ValueError("Too few paired patient bootstrap samples")
        bounds = np.quantile(samples, [0.025, 0.975], axis=0)
        intervals[key] = {"ap": bounds[:, 0].tolist(), "auroc": bounds[:, 1].tolist(), "n": len(samples)}
    return intervals


def compare_eicu(new_paths, originals):
    saved = read_json(EICU / "summary.json")
    if saved["model_sha256"] != originals or saved["bootstrap_replicates"] != BOOTSTRAPS or saved["bootstrap_seed"] != SEED:
        raise ValueError("P8 models or bootstrapping settings differ")
    cohort = pd.read_parquet(EICU / "cohort.parquet")
    reference = pd.read_parquet(EICU / "predictions.parquet")
    frames = {mode: pd.read_parquet(EICU / f"features_{mode}.parquet") for mode in MODES}
    if not cohort.index.is_unique or not all(frame.index.equals(cohort.index) for frame in [reference, *frames.values()]):
        raise ValueError("eICU cohort, features and prediction keys differ")
    if len(cohort) != saved["stays"] or int(cohort.y.sum()) != saved["deaths"] or cohort.uniquepid.nunique() != saved["patients"]:
        raise ValueError("eICU cohort differs from P8")
    paths = {model: FROZEN / f"{model}.joblib" for model in MODELS}
    if {model: sha(p) for model, p in paths.items()} != originals:
        raise ValueError("An original P8 model changed")
    print("Evaluating both original eICU cadence settings and paired patient intervals", flush=True)
    old, _, _ = score_frozen(frames, frames["hourly"], paths)
    original = {(row["mode"], row["model"]): row for row in saved["metrics"]}
    for key, values in old.items():
        np.testing.assert_array_equal(values, reference[key].to_numpy(dtype=values.dtype))
        baseline = original[tuple(key.split("_"))]
        measured = score_metrics(cohort.y.to_numpy(), values)
        if baseline["threshold"] != THRESHOLD:
            raise ValueError("P8 threshold differs")
        np.testing.assert_allclose([measured[m] for m in METRICS], [baseline[m] for m in METRICS], atol=1e-12, rtol=0)
    augmented, _, _ = score_frozen(frames, frames["hourly"], new_paths)
    result = evaluate_predictions(cohort, augmented)
    paired = paired_intervals(cohort, old, augmented)
    output = []
    for item in result:
        mode, model = item["mode"], item["model"]
        before = original[mode, model]
        row = {"mode": mode, "model": model}
        for metric in METRICS:
            row.update({
                f"{metric}_frozen": before[metric], f"{metric}_augmented": item[metric],
                f"delta_{metric}": item[metric] - before[metric],
            })
        for metric in ("ap", "auroc"):
            row[f"{metric}_augmented_ci"] = item[f"{metric}_ci"]
            row[f"delta_{metric}_paired_ci"] = paired[f"{mode}_{model}"][metric]
        row["paired_bootstraps"] = paired[f"{mode}_{model}"]["n"]
        output.append(row)
    return output


def run(models, output):
    output = output.resolve()
    root = OUT.resolve()
    if not output.is_relative_to(root) or output == root or output.exists() or not output.parent.is_dir():
        raise ValueError("Use a new output directory under ML/artifacts")
    model_paths, pipes, manifest = augmented_models(models)
    originals = manifest["original_model_sha256"]
    immutable = [models / "manifest.json", *model_paths.values(), OUT / "fit_ledger.jsonl", C_FEATURES, C_LABELS, C_PREDICTIONS, COVERAGE]
    immutable += [FROZEN / f"{name}.joblib" for name in MODELS]
    immutable += [P9 / "summary.json", P9 / "replicate_metrics.csv"]
    immutable += [EICU / file for file in ("cohort.parquet", "predictions.parquet", "summary.json", *(f"features_{mode}.parquet" for mode in MODES))]
    before = {path: sha(path) for path in immutable}

    with threadpool_limits(limits=2):
        set_c, replicates = compare_set_c(pipes, originals)
        eicu = compare_eicu(model_paths, originals)
    if any(sha(path) != expected for path, expected in before.items()):
        raise RuntimeError("An original input or model changed during evaluation")
    summary = {
        "training": manifest, "evaluation_fits": 0, "device": "cpu", "cpu_threads": 2,
        "set_c": set_c, "eicu": eicu,
    }
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="augmentation-eval-") as temporary:
        staged = Path(temporary).resolve() / "run"
        if staged.parent.parent != output.parent:
            raise RuntimeError("Unexpected temporary output path")
        staged.mkdir()
        pd.DataFrame(set_c).to_csv(staged / "set_c_comparison.csv", index=False)
        replicates.to_csv(staged / "set_c_paired_replicates.csv", index=False)
        pd.DataFrame(eicu).to_csv(staged / "eicu_comparison.csv", index=False)
        (staged / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(output)
    print(f"Saved paired aggregate comparisons under {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.models, args.output)
