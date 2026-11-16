"""Fit fixed P26 branches with real eICU Demo stays and evaluate held-out data."""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from audit_eicu_hospital_variation import hospital_blocks
from eicu_demo_features import DEMO
from evaluate_eicu_confirmation import frozen_ensemble, frozen_models, mapped_inputs
from evaluate_eicu_demo import source_hashes
from evaluate_gcs_routed_temporal import RoutedPredictor, checked_metrics
from evaluate_ma_gbt_p12 import P11, REFERENCES
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import benchmarks, compare_c, no_category_warning
from evaluate_robust_ensemble import EVALUATION as P28, Ensemble
from evaluate_temporal_features import eicu_temporal, save_json, set_c_temporal
from pilot_temporal_features import DEST as P22
from prepare import OUT, sha
from stress_missingness_c import METRICS, load_cohort, score_metrics
from train_gcs_routed_temporal import gcs_full, matrix, train_one
from train_predictive_boosting import SHIFT_SEED, ab_inputs, from_json, verify_sources

OUTPUT = OUT / "p44_real_eicu_augmentation"
SEED = 2026111744
P26_SEED = SHIFT_SEED + 400
BRANCHES = ("temporal", "blind_lgb")
NEW = "real_eicu_augmentation"
N_FOLDS = 5
HOSPITAL_SHA256 = "4ed356f9c4e7c8ce80e89241697e94b77f21c2f325cd6ad126794b1e92805ef4"


def partition(cohort):
    _, block_rows, _, _, counts = hospital_blocks(cohort)
    codes = np.full(len(cohort), -1, dtype=int)
    for index, block in enumerate(block_rows):
        codes[block] = index
    if (codes < 0).any():
        raise ValueError("Missing hospital grouping")
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    labels = cohort.y.to_numpy()
    covered = np.zeros(len(cohort), dtype=int)
    folds, sizes = [], []
    for train, tested in splitter.split(np.zeros(len(labels)), labels, groups=codes):
        if (
            set(cohort.uniquepid.iloc[train]).intersection(cohort.uniquepid.iloc[tested])
            or np.intersect1d(codes[train], codes[tested]).size
            or set(labels[train]) != {0, 1} or set(labels[tested]) != {0, 1}
        ):
            raise ValueError("Grouped eICU evaluation fold has overlapping keys or incomplete outcomes")
        covered[tested] += 1
        folds.append((train, tested))
        sizes.append({
            "train_stays": len(train), "test_stays": len(tested), "test_deaths": int(labels[tested].sum()),
            "test_hospital_blocks": len(np.unique(codes[tested])),
        })
    np.testing.assert_array_equal(covered, np.ones(len(labels), dtype=int))
    return folds, sizes, counts


def provenance(cohort):
    patients = pd.read_csv(DEMO / "patient.csv.gz", usecols=["hospitalid"])
    path = DEMO / "hospital.csv.gz"
    if sha(path) != HOSPITAL_SHA256:
        raise ValueError("Released Demo hospital file differs from the verified version")
    hospitals = pd.read_csv(path, usecols=["hospitalid"])["hospitalid"]
    if not hospitals.is_unique or hospitals.isna().any() or set(hospitals) != set(patients.hospitalid):
        raise ValueError("The released patient and hospital identifier sets differ")
    return {
        "released_patient_hospitalids": int(patients.hospitalid.nunique()),
        "released_hospital_table_hospitalids": len(hospitals),
        "eligible_hospitalids": int(cohort.hospitalid.nunique()),
        "hospital_sha256": HOSPITAL_SHA256,
    }


def train_pair(x, timing, y, extra, selected, y_demo):
    return {
        tag: train_one(tag, x, timing, y, P26_SEED, extra=(extra[tag][selected], y_demo[selected]))
        for tag in BRANCHES
    }


def external(x, timing, y, routed, gcs, rule, elastic, cols, extras, prior, old38):
    (cohort, original), hashes = eicu_temporal()
    if hashes != prior["eicu_raw_sha256"] or hashes != old38["eicu_raw_sha256"]:
        raise ValueError("Expected the original Demo sources and cohort")
    sources = provenance(cohort)
    frames = {("original", mode): frame for mode, frame in original.items()}
    mapped = {mode: frame.copy() for mode, frame in original.items()}
    mapped_inputs(DEMO, cohort, mapped)
    frames.update({("urine_mapped", mode): frame for mode, frame in mapped.items()})
    y_demo = cohort.y.to_numpy()
    demo = mapped["hourly"]
    features = demo.loc[:, list(x.columns)]
    time_features = demo.loc[:, extras]
    extra = {
        "temporal": matrix(features, features, time_features),
        "blind_lgb": matrix(features, gcs_full(features, P26_SEED), time_features),
    }
    folds, fold_sizes, grouping = partition(cohort)
    counts = len(cohort), int(y_demo.sum()), grouping["patients"]
    if counts != (782, 91, 719):
        raise ValueError("Unexpected eICU Demo population")
    previous, predictions, original_elastic = {}, {}, {}
    anchor = Ensemble(routed, gcs, rule)
    old = {(row["alignment"], row["mode"]): row["metrics"] for row in prior["eicu"]}
    ref = {(row["alignment"], row["mode"]): row["frozen_elastic"] for row in old38["eicu"]}
    for key, frame in frames.items():
        name = ":".join(key)
        previous[name] = anchor.predict_proba(frame)[:, 1]
        original_elastic[name] = no_category_warning(elastic, frame.loc[:, cols])
        checked_metrics(y_demo, previous[name], old[key])
        checked_metrics(y_demo, original_elastic[name], ref[key])
        predictions[name] = np.full(len(y_demo), np.nan)
    for fold, (train, tested) in enumerate(folds):
        components = train_pair(x, timing, y, extra, train, y_demo)
        new = Ensemble(RoutedPredictor(routed.rule, {**routed.models, **components}), gcs, rule)
        for key, frame in frames.items():
            predictions[":".join(key)][tested] = new.predict_proba(frame.iloc[tested])[:, 1]
        print(f"P44 fold {fold + 1}/{N_FOLDS}: augmented both branches with {len(train)} training Demo stays", flush=True)
    ci = paired_intervals(cohort, previous, predictions)
    ci_elastic = paired_intervals(cohort, original_elastic, predictions)
    rows = {
        key: {
            "metrics": score_metrics(y_demo, score),
            "p28": score_metrics(y_demo, previous[key]),
            "frozen_elastic": score_metrics(y_demo, original_elastic[key]),
            "paired_ci_vs_p28": ci[key], "paired_ci_vs_frozen_elastic": ci_elastic[key],
        } for key, score in predictions.items()
    }
    return rows, extra, y_demo, hashes, grouping, fold_sizes, sources


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    routed, _, gcs, extras, protected = frozen_models()
    rule, elastic, reduced = frozen_ensemble(protected)
    if routed.rule != {"primary": "temporal", "fallback": "blind_lgb", "weight": 1.0}:
        raise ValueError("Expected the selected P26 branch structure")
    prior_path, old38_path = P28 / "summary.json", OUT / "p38_sequence_channel_ablation/evaluation/summary.json"
    timing_path = P22 / "features_AB_temporal.parquet"
    p11_path, reference_path = P11 / "summary.json", REFERENCES / "manifest.json"
    prior, old38, p11, reference = map(from_json, (prior_path, old38_path, p11_path, reference_path))
    protected.update(prior["source_sha256"])
    protected.update({
        path.relative_to(OUT).as_posix(): sha(path)
        for path in (prior_path, old38_path, timing_path, p11_path, reference_path)
    })
    verify_sources(protected)
    x, y, _ = ab_inputs()
    timing = pd.read_parquet(timing_path)
    if not timing.index.equals(x.index) or timing.columns.tolist() != extras:
        raise ValueError("A/B timing schema changed")
    eicu, extra, y_demo, hashes, grouping, fold_sizes, identifier_sources = external(
        x, timing, y, routed, gcs, rule, elastic, reduced, extras, prior, old38,
    )
    fitted = train_pair(x, timing, y, extra, np.arange(len(y_demo)), y_demo)
    print("P44 final Set C models fitted on all A/B and all eligible Demo training stays", flush=True)
    raw, y_c, _ = load_cohort()
    times = set_c_temporal(raw.index)
    anchor = Ensemble(RoutedPredictor(routed.rule, routed.models, raw, times), gcs, rule)
    adapted = Ensemble(RoutedPredictor(routed.rule, {**routed.models, **fitted}, raw, times), gcs, rule)
    benchmark = {row["scenario"]: row for row in prior["set_c"]}
    checked_metrics(y_c, anchor.predict_proba(raw)[:, 1], benchmark["clean"])
    cols = x.columns.tolist()
    summary, masks, c_ci = compare_c(
        {NEW: (adapted, cols)}, {**benchmarks(), "p28": (anchor, cols)}, reference, p11,
    )
    for row in summary:
        for metric in METRICS:
            before = benchmark[row["scenario"]][metric]
            row[f"{metric}_p28"] = before
            row[f"delta_{metric}_vs_p28"] = row[metric] - before
    if hashes != source_hashes(DEMO) or sha(DEMO / "hospital.csv.gz") != HOSPITAL_SHA256:
        raise ValueError("Demo source changed during evaluation")
    verify_sources(protected)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="real-augmentation-") as tmp:
        stage = Path(tmp) / OUTPUT.name
        stage.mkdir()
        model_dir = stage / "fitted"
        model_dir.mkdir()
        for tag, model in fitted.items():
            joblib.dump(model, model_dir / f"{tag}.joblib")
        masks.to_csv(stage / "set_c_paired_replicates.csv", index=False)
        save_json(stage / "summary.json", {
            "device": "cpu", "ab_stays": len(y), "demo_stays": len(y_demo), "demo_deaths": int(y_demo.sum()),
            "extra_weight_per_stay": 1.0, "fold_grouping": "hospitalid_connected_components",
            "seed": SEED, "original_p26_seed": P26_SEED, "folds": fold_sizes, "grouping": grouping,
            "crossfit_branch_fits": len(BRANCHES) * len(fold_sizes), "final_branch_fits": len(fitted),
            "p26_rule": routed.rule, "p28_rule": rule, "model_sha256": {
                tag: sha(model_dir / f"{tag}.joblib") for tag in fitted
            },
            "demo_identifier_sources": identifier_sources, "eicu": eicu, "set_c": summary,
            "set_c_clean_ci": c_ci, "source_sha256": protected, "raw_eicu_sha256": hashes,
        })
        stage.rename(OUTPUT)
    print(f"P44 real eICU augmentation evaluated at {OUTPUT}", flush=True)


if __name__ == "__main__":
    evaluate()
