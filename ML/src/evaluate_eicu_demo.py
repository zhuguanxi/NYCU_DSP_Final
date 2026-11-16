"""Evaluate the saved P12 XGBoost pipelines on eICU Demo using CPU only."""

import argparse
import json
import os
import platform
import tempfile
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "2"

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits

from eicu_demo_features import DEMO, INPUT_TABLES, MODES, UNMAPPED, build_features
from prepare import OUT, STATIC, UNITS, sha


MODELS = ("Full", "Elastic")
FROZEN = OUT / "final_models"
P7_PROTOCOL = OUT / "C_evaluation/evaluation_protocol.json"
DEFAULT_OUTPUT = OUT / "eicu_demo_external"
THRESHOLD = 0.5
BOOTSTRAPS = 1000
SEED = 42


def source_hashes(directory):
    expected = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            digest, filename = line.split(maxsplit=1)
            expected[filename.lstrip("*").removeprefix("./")] = digest
    result = {}
    for table in INPUT_TABLES:
        filename = f"{table}.csv.gz"
        digest = sha(directory / filename)
        if digest != expected.get(filename):
            raise ValueError(f"eICU input failed checksum verification: {filename}")
        result[filename] = digest
    return result


def score_frozen(features, reference, model_paths):
    metadata = json.loads((FROZEN / "metadata.json").read_text(encoding="utf-8"))
    predictions = {}
    dimensions = {}
    unknown_category_warnings = {}
    for name in MODELS:
        pipe, _, columns = joblib.load(model_paths[name])
        model = pipe.named_steps["model"]
        if not isinstance(model, xgboost.XGBClassifier) or pipe.classes_.tolist() != [0, 1]:
            raise TypeError(f"Unexpected estimator or class ordering for {name}")
        if list(columns) != metadata[name]["columns"] or len(set(columns)) != len(columns):
            raise ValueError(f"Frozen predictor columns differ from metadata for {name}")
        if not set(columns).issubset(reference.columns):
            raise ValueError(f"Frozen predictor columns differ from the P12 reference for {name}")
        dimensions[name] = len(columns)
        model.set_params(device="cpu", n_jobs=2)
        if model.get_xgb_params().get("device") != "cpu":
            raise ValueError(f"CPU selection failed for {name}")
        with threadpool_limits(limits=2):
            for mode in MODES:
                frame = features[mode].loc[:, columns]
                with warnings.catch_warnings(record=True) as messages:
                    warnings.filterwarnings("always", message="Found unknown categories.*")
                    values = pipe.predict_proba(frame)[:, 1]
                for item in messages:
                    if "Found unknown categories" not in str(item.message):
                        warnings.warn(str(item.message), category=item.category, stacklevel=2)
                if len(values) != len(frame) or not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
                    raise ValueError(f"Invalid predictions for {name} ({mode})")
                key = f"{mode}_{name}"
                predictions[key] = values
                unknown_category_warnings[key] = sum("Found unknown categories" in str(item.message) for item in messages)
    return predictions, dimensions, unknown_category_warnings


def ranking(y, score):
    return average_precision_score(y, score), roc_auc_score(y, score)


def evaluate_predictions(cohort, predictions):
    y = cohort.y.to_numpy()
    groups = list(cohort.groupby("uniquepid", sort=False).indices.values())
    rng = np.random.default_rng(SEED)
    bootstrap = {key: [] for key in predictions}
    for _ in range(BOOTSTRAPS):
        ids = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in ids])
        targets = y[indices]
        if targets.min() == targets.max():
            continue
        for key, score in predictions.items():
            bootstrap[key].append(ranking(targets, score[indices]))
    if min(map(len, bootstrap.values())) < 0.95 * BOOTSTRAPS:
        raise ValueError("Too few valid patient bootstrap replicates")

    rows = []
    for mode in MODES:
        for name in MODELS:
            key = f"{mode}_{name}"
            score = predictions[key]
            ap, auroc = ranking(y, score)
            bounds = np.quantile(bootstrap[key], [0.025, 0.975], axis=0)
            tn, fp, fn, tp = map(int, confusion_matrix(y, score >= THRESHOLD, labels=[0, 1]).ravel())
            rows.append({
                "mode": mode, "model": name, "ap": float(ap), "auroc": float(auroc),
                "ap_ci": bounds[:, 0].tolist(), "auroc_ci": bounds[:, 1].tolist(),
                "brier": brier_score_loss(y, score), "mean_score": float(score.mean()),
                "threshold": THRESHOLD, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                "ppv": tp / (tp + fp) if tp + fp else None,
                "valid_bootstraps": len(bootstrap[key]),
            })
    return rows


def feature_coverage(reference, features):
    rows = []
    cases = {"p12_ab_adult": reference.loc[reference.static_Age.ge(18)], **features}
    for label, frame in cases.items():
        for variable in STATIC:
            present = frame[f"static_{variable}"].notna()
            rows.append((label, f"static_{variable}", len(frame), int(present.sum()), float(present.mean()), np.nan))
        for variable in UNITS:
            counts = frame[f"{variable}__count"]
            observed = counts[counts.gt(0)]
            rows.append((label, variable, len(frame), len(observed), len(observed) / len(frame), observed.median()))
    return pd.DataFrame(
        rows,
        columns=["cohort", "variable", "n", "observed_n", "observed_rate", "median_count_when_observed"],
    )


def run(directory, output):
    output = output.resolve()
    if not output.is_relative_to(OUT.resolve()) or output == OUT.resolve() or output.exists():
        raise ValueError("Output must be a new directory under ML/artifacts")
    if not output.parent.is_dir():
        raise ValueError("Output parent does not exist")

    model_paths = {name: FROZEN / f"{name}.joblib" for name in MODELS}
    model_hashes = {name: sha(path) for name, path in model_paths.items()}
    previous = json.loads(P7_PROTOCOL.read_text(encoding="utf-8"))
    if model_hashes != previous["model_sha256"]:
        raise ValueError("The saved models do not match the original P12 C evaluation")
    ledger = OUT / "fit_ledger.jsonl"
    ledger_sha = sha(ledger)
    data_hashes = source_hashes(directory)
    reference = pd.read_parquet(OUT / "features_AB.parquet")
    print("Building eICU Demo features on CPU", flush=True)
    cohort, features, flow, dropped_late = build_features(directory)
    if any(set(frame.columns) != set(reference.columns) for frame in features.values()):
        raise ValueError("The generated feature names differ from original P12")
    print(f"Scoring {len(cohort)} eligible stays with both frozen CPU pipelines", flush=True)
    predictions, dimensions, category_warnings = score_frozen(features, reference, model_paths)
    metrics = evaluate_predictions(cohort, predictions)
    prevalence = float(cohort.y.mean())
    summary = {
        "dataset": "eicu-crd-demo-2.0.1", "device": "cpu", "cpu_threads": 2,
        "outcome": "hospitaldischargestatus Expired versus Alive",
        "flow": flow, "stays": len(cohort), "patients": cohort.uniquepid.nunique(),
        "hospital_ids": cohort.hospitalid.nunique(), "deaths": int(cohort.y.sum()),
        "mortality_rate": prevalence, "prevalence_only_ap": prevalence,
        "prevalence_only_brier": prevalence * (1 - prevalence),
        "unmapped_dynamic_variables": sorted(UNMAPPED), "frozen_input_counts": dimensions,
        "observations_after_cutoff_by_entry": dropped_late,
        "data_sha256": data_hashes, "model_sha256": model_hashes,
        "unknown_category_warnings": category_warnings, "added_training_fits": 0,
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": SEED,
        "python": platform.python_version(), "xgboost": xgboost.__version__, "metrics": metrics,
    }

    if any(sha(path) != model_hashes[name] for name, path in model_paths.items()) or sha(ledger) != ledger_sha:
        raise RuntimeError("The frozen artifacts or fit ledger changed during external evaluation")
    coverage = feature_coverage(reference, features)
    with tempfile.TemporaryDirectory(prefix="eicu-validation-", dir=output.parent) as temporary:
        staging = Path(temporary).resolve() / "run"
        if staging.parent.parent != output.parent:
            raise RuntimeError("Unexpected staging directory")
        staging.mkdir()
        for mode, frame in features.items():
            frame.to_parquet(staging / f"features_{mode}.parquet")
        cohort.to_parquet(staging / "cohort.parquet")
        pd.DataFrame(predictions, index=cohort.index).to_parquet(staging / "predictions.parquet")
        coverage.to_csv(staging / "feature_coverage.csv", index=False)
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(output)

    for row in metrics:
        print(f"{row['mode']} {row['model']}: AP {row['ap']:.4f}; AUROC {row['auroc']:.4f}; Brier {row['brier']:.4f}")
    print(f"Saved local outputs under {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEMO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.dataset, args.output)
