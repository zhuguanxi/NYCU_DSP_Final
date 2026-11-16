"""Combine the frozen window model with the exact P23 urine alignment."""

import json
import tempfile
from pathlib import Path

from evaluate_temporal_features import EVALUATION as P22_EVAL
from evaluate_temporal_features import FITTED, TemporalPredictor, eicu_temporal, save_json

import joblib
import numpy as np
import pandas as pd

from eicu_demo_features import DEMO, STAY
from evaluate_eicu_demo import MODES, score_frozen, source_hashes
from evaluate_eicu_urine_mapping import OUTPUT as P23, urine_events, urine_values
from evaluate_missingness_augmentation import augmented_models, paired_intervals
from pilot_temporal_features import ID, MINUTE, VALUE, time_summary
from prepare import OUT, sha
from stress_missingness_c import METRICS, score_metrics
from train_predictive_boosting import AB_FEATURES, FROZEN, GCS_ONLY_OUTPUT, verify_sources
from train_missingness_augmentation import MODELS, from_json

OUTPUT = OUT / "p24_eicu_temporal_urine"


def mapped_elastic_controls(frames, y, p23):
    original_paths = {name: FROZEN / f"{name}.joblib" for name in MODELS}
    gcs_paths, _, _ = augmented_models(GCS_ONLY_OUTPUT)
    p23_rows = {(row["variant"], row["model"], row["mode"]): row for row in p23["rows"]}
    output = {}
    for tag, paths in (("frozen", original_paths), ("gcs", gcs_paths)):
        scores, _, _ = score_frozen(frames, frames["hourly"], paths)
        output[f"{tag}_elastic_mapped"] = {mode: scores[f"{mode}_Elastic"] for mode in MODES}
        for mode, probabilities in output[f"{tag}_elastic_mapped"].items():
            measured = score_metrics(y, probabilities)
            expected = p23_rows[tag, "Elastic", mode]["urine_mapped"]
            np.testing.assert_allclose([measured[m] for m in METRICS], [expected[m] for m in METRICS], atol=1e-12, rtol=0)
    return output


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p22_path, p23_path = P22_EVAL / "summary.json", P23 / "summary.json"
    p22, p23 = from_json(p22_path), from_json(p23_path)
    model_path = FITTED / "xgboost.joblib"
    fitted = from_json(FITTED / "manifest.json")
    if p22["fitted"] != fitted or fitted["model_sha256"] != sha(model_path) or fitted["device"] != "cpu":
        raise ValueError("Temporal fitted model differs from P22")
    protected = {
        **p22["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in (p22_path, p23_path)},
    }
    verify_sources(protected)
    (cohort, frames), raw_sources = eicu_temporal()
    if raw_sources != p23["raw_sha256"]:
        raise ValueError("P23 urine mapping used different raw sources")
    y = cohort.y.to_numpy()
    events, coverage = urine_events(cohort.index)
    if coverage != p23["coverage"]:
        raise ValueError("Urine event mapping differs from P23")
    urine = urine_values(cohort.index, events)
    urine_data = events.rename(columns={STAY: ID, "minute": MINUTE, "value": VALUE})
    timing = time_summary(urine_data, cohort.index).filter(like="temporal_Urine__")
    model = TemporalPredictor(joblib.load(model_path), fitted["selected_rounds"])
    raw = pd.read_parquet(AB_FEATURES).columns.tolist()
    columns = raw + fitted["pilot"]["features"]
    swapped = {}
    old, next_score = {}, {}
    for mode, frame in frames.items():
        old[mode] = model.predict_proba(frame.loc[:, columns])[:, 1]
        updated = frame.copy()
        updated.loc[:, urine.columns] = urine
        updated.loc[:, timing.columns] = timing
        next_score[mode] = model.predict_proba(updated.loc[:, columns])[:, 1]
        swapped[mode] = updated.loc[:, raw]

    controls = {"temporal_unmapped": old, **mapped_elastic_controls(swapped, y, p23)}
    p22_rows = {row["mode"]: row for row in p22["eicu"]}
    for mode in MODES:
        np.testing.assert_allclose(
            [score_metrics(y, old[mode])[m] for m in METRICS],
            [p22_rows[mode][m] for m in METRICS], atol=1e-12, rtol=0,
        )
    intervals = {tag: paired_intervals(cohort, scores, next_score) for tag, scores in controls.items()}
    results = []
    for mode in MODES:
        actual = score_metrics(y, next_score[mode])
        row = {"mode": mode, "temporal_mapped": actual, "comparison": {}}
        for tag, scores in controls.items():
            before = score_metrics(y, scores[mode])
            row["comparison"][tag] = {
                "metrics": before, "delta": {key: actual[key] - before[key] for key in METRICS},
                "paired_ci": intervals[tag][mode],
            }
        results.append(row)
    if source_hashes(DEMO) != raw_sources:
        raise ValueError("Raw eICU files changed during the paired evaluation")
    verify_sources(protected)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p24-combined-") as temporary:
        staging = Path(temporary) / OUTPUT.name
        staging.mkdir()
        save_json(staging / "summary.json", {
            "device": "cpu", "fits": 0, "coverage": coverage, "rows": results,
            "source_sha256": protected, "eicu_raw_sha256": raw_sources,
        })
        staging.rename(OUTPUT)
    print(f"Combined temporal and urine mapping evaluation saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    run()
