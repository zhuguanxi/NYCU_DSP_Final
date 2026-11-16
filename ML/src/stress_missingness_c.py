"""Stress-test frozen P12 Set C predictions with eICU-matched channel masking."""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "2"

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import brier_score_loss
from threadpoolctl import threadpool_limits

from evaluate_eicu_demo import FROZEN, MODELS, P7_PROTOCOL, THRESHOLD, ranking
from prepare import OUT, sha, stats_spec


C = OUT / "C_evaluation"
C_FEATURES = C / "features_C.parquet"
C_LABELS = C / "labels_C.csv"
C_PREDICTIONS = C / "predictions_C.csv"
C_MANIFEST = C / "evaluation_manifest.json"
COVERAGE = OUT / "eicu_demo_external/feature_coverage.csv"
DEFAULT_OUTPUT = OUT / "missingness_stress_C"
PARTIAL = ("GCS", "Lactate", "MAP")
UNMAPPED = ("Mg", "Urine", "MechVent")
CHANNELS = PARTIAL + UNMAPPED
REPEATS = 32
SEED = 20261116
SCENARIOS = (
    *((f"matched_{name}", (name,), "matched") for name in PARTIAL),
    *((f"full_{name}", (name,), "full") for name in UNMAPPED),
    ("matched_aligned_three", PARTIAL, "matched"),
    ("full_unmapped_three", UNMAPPED, "full"),
    ("matched_all_six", CHANNELS, "matched"),
    *((f"full_{name}", (name,), "full") for name in PARTIAL),
    ("full_all_six", CHANNELS, "full"),
)
METRICS = ("ap", "auroc", "brier", "sensitivity")


def load_cohort():
    features = pd.read_parquet(C_FEATURES)
    predictions = pd.read_csv(C_PREDICTIONS).set_index("RecordID")
    outcome = pd.read_csv(C_LABELS).set_index("RecordID")["In_hospital_death"]
    if features.shape != (4000, 435) or not features.index.is_unique:
        raise ValueError("Expected the original 4000-stay Set C feature table")
    if not features.index.equals(outcome.index) or not features.index.equals(predictions.index):
        raise ValueError("Set C feature, label and prediction orders differ")
    y = outcome.to_numpy()
    if set(y) != {0, 1} or int(y.sum()) != 585 or not np.array_equal(y, predictions.y):
        raise ValueError("Set C mortality labels differ from the original evaluation")
    source_hash = json.loads(C_MANIFEST.read_text(encoding="utf-8"))["C_feature_sha256"]
    if sha(C_FEATURES) != source_hash:
        raise ValueError("Set C feature checksum differs from the original evaluation")
    return features, y, predictions


def load_pipelines(features):
    paths = {name: FROZEN / f"{name}.joblib" for name in MODELS}
    frozen = json.loads(P7_PROTOCOL.read_text(encoding="utf-8"))["model_sha256"]
    if {name: sha(path) for name, path in paths.items()} != frozen:
        raise ValueError("A frozen model checksum differs from the original evaluation")
    meta = json.loads((FROZEN / "metadata.json").read_text(encoding="utf-8"))
    pipelines = {}
    for name, path in paths.items():
        pipeline, _, columns = joblib.load(path)
        model = pipeline.named_steps["model"]
        if not isinstance(model, xgboost.XGBClassifier) or pipeline.classes_.tolist() != [0, 1]:
            raise TypeError(f"Unexpected frozen estimator for {name}")
        if list(columns) != meta[name]["columns"] or not set(columns) <= set(features.columns):
            raise ValueError(f"Frozen feature schema differs for {name}")
        model.set_params(device="cpu", n_jobs=2)
        if model.get_xgb_params().get("device") != "cpu":
            raise RuntimeError(f"Could not set {name} to CPU")
        pipelines[name] = (pipeline, columns)
    return pipelines, frozen


def mask_plan(features):
    coverage = pd.read_csv(COVERAGE)
    selected = coverage.loc[coverage.cohort.eq("hourly") & coverage.variable.isin(CHANNELS)].set_index("variable")
    if not selected.index.is_unique or set(selected.index) != set(CHANNELS):
        raise ValueError("Expected one eICU coverage entry per target channel")
    positions = {}
    plan = {}
    for variable in CHANNELS:
        row = selected.loc[variable]
        observed = int(row.observed_n)
        total = int(row.n)
        if not 0 <= observed <= total or total <= 0 or (variable in UNMAPPED and observed != 0):
            raise ValueError(f"Unexpected eICU coverage for {variable}")
        current = np.flatnonzero(features[f"{variable}__count"].gt(0).to_numpy())
        target = (2 * observed * len(features) + total) // (2 * total)
        retained = min(len(current), target)
        positions[variable] = current
        plan[variable] = {
            "eicu_observed": observed, "eicu_stays": total, "set_c_stays": len(features),
            "set_c_original_observed": len(current), "set_c_requested_observed": target,
            "set_c_matched_observed": retained, "matched_rows_to_mask": len(current) - retained,
        }
    return positions, plan


def mask_draws(positions, plan):
    rng = np.random.default_rng(SEED)
    return [
        {
            variable: rng.choice(positions[variable], size=plan[variable]["matched_rows_to_mask"], replace=False)
            for variable in PARTIAL
        }
        for _ in range(REPEATS)
    ]


def apply_masks(features, masks, plan, design):
    masked = features.copy()
    for variable, rows in masks.items():
        index = masked.index[rows]
        specs = stats_spec(variable)
        missing_values = [f"{variable}__{name}" for name in specs]
        zero_columns = [f"{variable}__count"]
        zero_columns.extend(f"{variable}__{name}_defined" for name in ("std", "slope") if name in specs)
        masked.loc[index, missing_values] = np.nan
        masked.loc[index, zero_columns] = 0
        masked.loc[index, f"{variable}__missing"] = 1
        retained = plan[variable]["set_c_matched_observed"] if design == "matched" else 0
        if int(masked[f"{variable}__count"].gt(0).sum()) != retained:
            raise RuntimeError(f"The mask has unexpected coverage for {variable}")
    return masked


def score_metrics(y, score):
    if not (np.isfinite(score) & (score >= 0) & (score <= 1)).all():
        raise ValueError("Invalid predicted probabilities")
    ap, auroc = ranking(y, score)
    return {
        "ap": float(ap), "auroc": float(auroc), "brier": float(brier_score_loss(y, score)),
        "sensitivity": float(np.mean(score[y == 1] >= THRESHOLD)),
    }


def stress(features, y, historical, pipelines, positions, plan):
    baseline = {}
    rows = []
    draws = mask_draws(positions, plan)
    with threadpool_limits(limits=2):
        for model, (pipeline, columns) in pipelines.items():
            probability = pipeline.predict_proba(features.loc[:, columns])[:, 1]
            np.testing.assert_array_equal(probability, historical[f"score_{model}"].to_numpy(dtype=probability.dtype))
            baseline[model] = score_metrics(y, probability)
        print("Unmasked Set C CPU predictions reproduce both saved models", flush=True)

        for name, variables, design in SCENARIOS:
            repeats = draws if design == "matched" else [{}]
            for repeat, sample in enumerate(repeats, start=1):
                masks = {variable: sample.get(variable, positions[variable]) for variable in variables}
                masked = apply_masks(features, masks, plan, design)
                for model, (pipeline, columns) in pipelines.items():
                    score = pipeline.predict_proba(masked.loc[:, columns])[:, 1]
                    metrics = score_metrics(y, score)
                    delta = {f"delta_{key}": value - baseline[model][key] for key, value in metrics.items()}
                    rows.append({
                        "scenario": name, "design": design, "repeat": repeat, "model": model,
                        "variables": "+".join(variables), **metrics, **delta,
                    })
            print(f"Completed {name} ({len(repeats)} mask{'s' if len(repeats) != 1 else ''})", flush=True)
    return baseline, pd.DataFrame(rows)


def summarize(replicates):
    summary = []
    keys = ["scenario", "model", "variables", "design"]
    for (scenario, model, variables, design), rows in replicates.groupby(keys, sort=False):
        result = {"scenario": scenario, "model": model, "variables": variables, "design": design, "repeats": len(rows)}
        for metric in METRICS:
            result[metric] = float(rows[metric].mean())
            delta = rows[f"delta_{metric}"]
            result[f"delta_{metric}"] = float(delta.mean())
            result[f"delta_{metric}_p10_p90"] = delta.quantile([0.1, 0.9]).tolist()
        summary.append(result)
    return summary


def run(output):
    output = output.resolve()
    root = OUT.resolve()
    if output == root or not output.is_relative_to(root) or output.exists() or not output.parent.is_dir():
        raise ValueError("Use a new output directory under ML/artifacts")
    untouched = [C_FEATURES, C_LABELS, C_PREDICTIONS, COVERAGE, OUT / "fit_ledger.jsonl"]
    untouched.extend(FROZEN / f"{name}.joblib" for name in MODELS)
    hashes = {path: sha(path) for path in untouched}
    features, y, historical = load_cohort()
    pipelines, model_hashes = load_pipelines(features)
    positions, plan = mask_plan(features)
    baseline, replicates = stress(features, y, historical, pipelines, positions, plan)
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise RuntimeError("A source artifact or the fit ledger changed during the stress test")

    report = {
        "cohort": "P12 Set C", "stays": len(features), "deaths": int(y.sum()), "device": "cpu", "cpu_threads": 2,
        "threshold": THRESHOLD, "seed": SEED, "matched_repeats": REPEATS, "added_training_fits": 0,
        "model_sha256": model_hashes, "source_sha256": {path.name: value for path, value in hashes.items()},
        "baseline": baseline, "mask_plan": plan, "scenarios": summarize(replicates),
    }
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="missingness-run-") as temporary:
        staging = Path(temporary).resolve() / "run"
        if staging.parent.parent != output.parent:
            raise RuntimeError("Unexpected staging directory")
        staging.mkdir()
        replicates.to_csv(staging / "replicate_metrics.csv", index=False)
        (staging / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(output)
    print(f"Verified the source and frozen model hashes. Results saved under {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args()
    run(args.output)
