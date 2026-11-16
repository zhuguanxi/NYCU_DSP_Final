"""Compare the urine-only ablation with the three frozen P9–P11 variants."""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from compare_gcs_augmentation import (
    DEFAULT_OUTPUT as P11_COMPARISON, DUAL_EVALUATION, DUAL_MODELS, GCS_EVALUATION, checked_run,
)
from evaluate_eicu_demo import BOOTSTRAPS, MODES, SEED, ranking, score_frozen
from evaluate_missingness_augmentation import EICU, METRICS, read_json
from prepare import OUT, sha
from stress_missingness_c import score_metrics
from train_missingness_augmentation import GCS_ONLY_OUTPUT, URINE_ONLY_OUTPUT


URINE_EVALUATION = OUT / "urine_only_augmentation_evaluation"
DEFAULT_OUTPUT = OUT / "urine_only_comparison"
PRIOR_VARIANTS = ("frozen", "gcs", "dual")
OTHER_AUGMENTATIONS = ("gcs", "dual")


def four_way(previous, current):
    result = {}
    for metric in METRICS:
        original = previous[f"{metric}_frozen"]
        if abs(original - current[f"{metric}_frozen"]) > 1e-12:
            raise ValueError("The original-model reference differs")
        urine = current[f"{metric}_augmented"]
        result[f"{metric}_urine"] = urine
        for version in PRIOR_VARIANTS:
            prior = previous[f"{metric}_{version}"]
            result[f"{metric}_{version}"] = prior
            result[f"delta_{metric}_vs_{version}"] = urine - prior
    return result


def compare_c(previous, current, evaluation):
    prior_draws = pd.read_csv(P11_COMPARISON / "set_c_three_way_replicates.csv")
    new_draws = pd.read_csv(evaluation / "set_c_paired_replicates.csv")
    keys = ["scenario", "model", "repeat", "design", "variables"]
    pairs = prior_draws.merge(new_draws, on=keys, suffixes=("_prior", "_current"), validate="one_to_one")
    if not len(pairs) == len(prior_draws) == len(new_draws) == 336:
        raise ValueError("The four models do not share all the original P9 mask draws")
    paired = pairs.loc[:, keys].copy()
    for metric in METRICS:
        np.testing.assert_allclose(pairs[f"{metric}_frozen_prior"], pairs[f"{metric}_frozen_current"], rtol=0, atol=1e-12)
        paired[f"{metric}_frozen"] = pairs[f"{metric}_frozen_prior"]
        paired[f"{metric}_urine"] = pairs[f"{metric}_augmented"]
        for version in OTHER_AUGMENTATIONS:
            paired[f"{metric}_{version}"] = pairs[f"{metric}_{version}"]
        for version in PRIOR_VARIANTS:
            paired[f"delta_{metric}_vs_{version}"] = paired[f"{metric}_urine"] - paired[f"{metric}_{version}"]

    previous_rows = {(row["scenario"], row["model"]): row for row in previous["set_c"]}
    output = []
    for row in current["set_c"]:
        key = (row["scenario"], row["model"])
        result = {"scenario": key[0], "model": key[1], "repeats": row["repeats"], **four_way(previous_rows[key], row)}
        if key[0] != "clean":
            matched = paired.loc[paired.scenario.eq(key[0]) & paired.model.eq(key[1])]
            if len(matched) != row["repeats"]:
                raise ValueError("Unexpected number of paired mask draws")
            for metric in METRICS:
                for version in PRIOR_VARIANTS:
                    name = f"delta_{metric}_vs_{version}"
                    changes = matched[name]
                    np.testing.assert_allclose(changes.mean(), result[name], rtol=0, atol=1e-12)
                    result[f"{name}_p10_p90"] = changes.quantile([0.1, 0.9]).tolist()
        output.append(result)
    if len(output) != len(previous_rows) or len(output) != 28:
        raise ValueError("Missing Set C scenarios")
    return output, paired


def paired_eicu_intervals(cohort, scores):
    groups = list(cohort.groupby("uniquepid", sort=False).indices.values())
    rng = np.random.default_rng(SEED)
    y = cohort.y.to_numpy()
    changes = {(key, version): [] for key in scores["urine"] for version in OTHER_AUGMENTATIONS}
    for _ in range(BOOTSTRAPS):
        chosen = rng.integers(len(groups), size=len(groups))
        rows = np.concatenate([groups[index] for index in chosen])
        truth = y[rows]
        if truth.min() == truth.max():
            continue
        for key, values in scores["urine"].items():
            new = ranking(truth, values[rows])
            for version in OTHER_AUGMENTATIONS:
                old = ranking(truth, scores[version][key][rows])
                changes[key, version].append(np.subtract(new, old))

    intervals = {}
    for key, samples in changes.items():
        if len(samples) != BOOTSTRAPS:
            raise ValueError("Expected the same valid eICU bootstrap draws as P8–P11")
        bounds = np.quantile(samples, [0.025, 0.975], axis=0)
        intervals[key] = {"ap": bounds[:, 0].tolist(), "auroc": bounds[:, 1].tolist()}
    return intervals


def compare_eicu(previous, current, model_paths):
    cohort = pd.read_parquet(EICU / "cohort.parquet")
    frames = {mode: pd.read_parquet(EICU / f"features_{mode}.parquet") for mode in MODES}
    if not cohort.index.is_unique or not all(frame.index.equals(cohort.index) for frame in frames.values()):
        raise ValueError("eICU inputs differ from the original cohort order")
    scores = {version: score_frozen(frames, frames["hourly"], paths)[0] for version, paths in model_paths.items()}
    prior = {(row["mode"], row["model"]): row for row in previous["eicu"]}
    output = []
    y = cohort.y.to_numpy()
    intervals = paired_eicu_intervals(cohort, scores)
    for row in current["eicu"]:
        key = (row["mode"], row["model"])
        prediction_key = "_".join(key)
        result = {"mode": key[0], "model": key[1], **four_way(prior[key], row)}
        for version, predictions in scores.items():
            metrics = score_metrics(y, predictions[prediction_key])
            np.testing.assert_allclose([metrics[m] for m in METRICS], [result[f"{m}_{version}"] for m in METRICS], rtol=0, atol=1e-12)
        for metric in ("ap", "auroc"):
            result[f"delta_{metric}_vs_frozen_ci"] = row[f"delta_{metric}_paired_ci"]
            for version in OTHER_AUGMENTATIONS:
                result[f"delta_{metric}_vs_{version}_ci"] = intervals[prediction_key, version][metric]
        result["paired_bootstraps"] = BOOTSTRAPS
        output.append(result)
    if len(output) != len(prior) or len(output) != 4:
        raise ValueError("Missing eICU comparisons")
    return output


def run(models, evaluation, output):
    output = output.resolve()
    root = OUT.resolve()
    if output == root or not output.is_relative_to(root) or output.exists() or not output.parent.is_dir():
        raise ValueError("Use a new output directory under ML/artifacts")
    previous = read_json(P11_COMPARISON / "summary.json")
    dual_paths, dual = checked_run(DUAL_MODELS, DUAL_EVALUATION, ("GCS", "Urine"))
    gcs_paths, gcs = checked_run(GCS_ONLY_OUTPUT, GCS_EVALUATION, ("GCS",))
    urine_paths, current = checked_run(models, evaluation, ("Urine",))
    if previous["gcs_training"] != gcs["training"]:
        raise ValueError("The existing comparison does not refer to these GCS models")
    for prior in (dual, gcs):
        if prior["training"]["original_model_sha256"] != current["training"]["original_model_sha256"]:
            raise ValueError("The original-model references differ")
        for name in ("features_AB.parquet", "labels_AB.csv", "folds3.csv"):
            if prior["training"]["source_sha256"][name] != current["training"]["source_sha256"][name]:
                raise ValueError(f"The training versions used different A/B inputs: {name}")

    model_paths = {"dual": dual_paths, "gcs": gcs_paths, "urine": urine_paths}
    protected = [path for version in model_paths.values() for path in version.values()]
    protected += [folder / "manifest.json" for folder in (DUAL_MODELS, GCS_ONLY_OUTPUT, models)]
    protected += [folder / "summary.json" for folder in (DUAL_EVALUATION, GCS_EVALUATION, P11_COMPARISON, evaluation)]
    protected += [P11_COMPARISON / "set_c_three_way_replicates.csv", evaluation / "set_c_paired_replicates.csv"]
    protected += [EICU / "cohort.parquet", *(EICU / f"features_{mode}.parquet" for mode in MODES)]
    hashes = {path: sha(path) for path in protected}

    set_c, pairs = compare_c(previous, current, evaluation)
    print("Verified four-way pairing of all 336 Set C comparisons", flush=True)
    print("Scoring previous and urine-only models for paired eICU intervals", flush=True)
    with threadpool_limits(limits=2):
        eicu = compare_eicu(previous, current, model_paths)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("A frozen model or comparison input changed")
    result = {
        "urine_training": current["training"], "comparison_device": "cpu", "comparison_fits": 0,
        "set_c": set_c, "eicu": eicu,
    }
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="urine-comparison-") as temporary:
        staged = Path(temporary).resolve() / "run"
        if staged.parent.parent != output.parent:
            raise RuntimeError("Unexpected temporary output")
        staged.mkdir()
        pd.DataFrame(set_c).to_csv(staged / "set_c_four_way.csv", index=False)
        pairs.to_csv(staged / "set_c_four_way_replicates.csv", index=False)
        pd.DataFrame(eicu).to_csv(staged / "eicu_four_way.csv", index=False)
        (staged / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(output)
    print(f"Saved four-way aggregate comparisons under {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=URINE_ONLY_OUTPUT)
    parser.add_argument("--evaluation", type=Path, default=URINE_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.models, args.evaluation, args.output)
