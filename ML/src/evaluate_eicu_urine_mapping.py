"""Test a prespecified eICU urine-volume mapping with frozen P12 models."""

import json
import os
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd

from eicu_demo_features import DEMO, STAY, chart_events, merge_minutes, normalize, numeric_events, windowed
from evaluate_eicu_demo import DEFAULT_OUTPUT, FROZEN, MODES, score_frozen, source_hashes
from evaluate_ma_gbt_p12 import REFERENCES, eicu_inputs
from evaluate_missingness_augmentation import augmented_models, paired_intervals
from prepare import OUT, extract_record, sha
from stress_missingness_c import METRICS, score_metrics
from train_missingness_augmentation import GCS_ONLY_OUTPUT, MODELS, from_json

OUTPUT = OUT / "p23_eicu_explicit_urine"
URINE_PATH = "flowsheet|flowsheet cell labels|i&o|output (ml)|urine"


def urine_events(index):
    frame, late = windowed(
        DEMO, "intakeOutput", index, "intakeoutputoffset",
        ("cellpath", "celllabel", "cellvaluenumeric"), "intakeoutputentryoffset",
    )
    select = normalize(frame.cellpath).eq(URINE_PATH) & normalize(frame.celllabel).eq("urine")
    raw = numeric_events(chart_events(frame, select, "Urine", "cellvaluenumeric"))
    events = merge_minutes(raw)
    return events, {
        "matched_numeric_rows": len(raw), "deduplicated_events": len(events),
        "observed_stays": events[STAY].nunique(), "cohort_stays": len(index), "all_io_rows_excluded_as_late": late,
    }


def urine_values(index, events):
    grouped = {key: subset for key, subset in events.groupby(STAY, sort=False)}
    rows = {}
    for stay in index:
        lines = ["Time,Parameter,Value", f"00:00,RecordID,{stay}"]
        if stay in grouped:
            lines.extend(
                f"{int(minute) // 60:02}:{int(minute) % 60:02},Urine,{value}"
                for minute, value in grouped[stay][["minute", "value"]].itertuples(index=False, name=None)
            )
        features, _, _ = extract_record("\n".join(lines).encode(), stay)
        rows[stay] = {key: value for key, value in features.items() if key.startswith("Urine__")}
    output = pd.DataFrame.from_dict(rows, orient="index").rename_axis(index.name)
    output = output.reindex(index)
    if not output.index.equals(index) or int(output["Urine__count"].sum()) != len(events):
        raise ValueError("Urine event alignment changed")
    return output


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reference = from_json(REFERENCES / "manifest.json")
    prior = from_json(DEFAULT_OUTPUT / "summary.json")
    raw_hashes = source_hashes(DEMO)
    if raw_hashes != prior["data_sha256"]:
        raise ValueError("Raw eICU files differ from the original frozen evaluation")
    original_paths = {name: FROZEN / f"{name}.joblib" for name in MODELS}
    if {name: sha(path) for name, path in original_paths.items()} != prior["model_sha256"]:
        raise ValueError("The original XGBoost models changed")
    gcs_paths, _, manifest = augmented_models(GCS_ONLY_OUTPUT)
    if manifest["selected_rates_percent"] != {"GCS": 60}:
        raise ValueError("Expected the frozen GCS-only models")
    cohort, frames = eicu_inputs()
    y = cohort.y.to_numpy()
    events, coverage = urine_events(cohort.index)
    urine = urine_values(cohort.index, events)
    remapped = {}
    for mode, frame in frames.items():
        if frame["Urine__count"].ne(0).any():
            raise ValueError("Expected the original unmapped urine feature")
        updated = frame.copy()
        updated.loc[:, urine.columns] = urine
        pd.testing.assert_frame_equal(updated.drop(columns=urine.columns), frame.drop(columns=urine.columns), check_exact=True)
        remapped[mode] = updated
    old_gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    if sha(old_gcs_path) != reference["gcs_scores_sha256"]:
        raise ValueError("Frozen GCS-only eICU scores changed")
    old_frame = {"frozen": pd.read_parquet(DEFAULT_OUTPUT / "predictions.parquet"), "gcs": pd.read_parquet(old_gcs_path)}
    results = []
    for tag, paths in (("frozen", original_paths), ("gcs", gcs_paths)):
        stored = old_frame[tag]
        if not stored.index.equals(cohort.index):
            raise ValueError("Old eICU predictions are misaligned")
        old, _, _ = score_frozen(frames, frames["hourly"], paths)
        new, _, _ = score_frozen(remapped, frames["hourly"], paths)
        for key, values in old.items():
            np.testing.assert_array_equal(values, stored[key].to_numpy(dtype=values.dtype))
        intervals = paired_intervals(cohort, old, new)
        for mode in MODES:
            for name in MODELS:
                key = f"{mode}_{name}"
                baseline = score_metrics(y, old[key])
                updated = score_metrics(y, new[key])
                results.append({
                    "variant": tag, "model": name, "mode": mode, "original": baseline, "urine_mapped": updated,
                    "difference": {metric: updated[metric] - baseline[metric] for metric in METRICS},
                    "paired_ci": intervals[key],
                })
    if source_hashes(DEMO) != raw_hashes:
        raise ValueError("Raw eICU files changed during evaluation")
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p23-urine-") as temporary:
        staging = Path(temporary) / OUTPUT.name
        staging.mkdir()
        (staging / "summary.json").write_text(json.dumps({
            "device": "cpu", "raw_sha256": raw_hashes, "path": URINE_PATH, "coverage": coverage,
            "cohort_deaths": int(y.sum()), "rows": results, "fits": 0,
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(OUTPUT)
    print(f"Outcome-independent urine mapping evaluated in {OUTPUT}", flush=True)


if __name__ == "__main__":
    run()
