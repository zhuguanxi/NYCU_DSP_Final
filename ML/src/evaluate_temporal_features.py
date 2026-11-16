"""Freeze the A/B temporal candidate and evaluate it on the original holdouts."""

import json
import tarfile
import tempfile
from pathlib import Path

from pilot_temporal_features import DEST, GROUPS, ID, MINUTE, VALUE, temporal_view, time_summary

import joblib
import numpy as np
import pandas as pd

from eicu_demo_features import DEMO, STAY, build_features
from evaluate_eicu_demo import DEFAULT_OUTPUT as EICU_SCORES, source_hashes
from evaluate_ma_gbt_p12 import P11, REFERENCES, eicu_inputs
from evaluate_predictive_boosting import C_BOOTSTRAPS, benchmarks, compare_c, compare_eicu
from prepare import DATA, OUT, extract_record, sha
from stress_missingness_c import load_cohort
from train_predictive_boosting import AB_FEATURES, ROUNDS, ab_inputs, estimator, native_inputs, native_predict
from train_predictive_boosting import verify_sources
from train_missingness_augmentation import from_json

FITTED = DEST / "fitted"
EVALUATION = DEST / "evaluation"
ARCHIVE = DATA / "set-c.tar.gz"
TAG = "temporal_selected"
SPEC = {"algorithm": "xgb3", "scope": "full", "augment": False}


class TemporalPredictor:
    def __init__(self, model, rounds, clean=None, temporal=None):
        self.model, self.rounds = model, rounds
        self.clean, self.temporal = clean, temporal

    def predict_proba(self, frame):
        if self.temporal is not None:
            temporal = temporal_view(self.clean.loc[frame.index], frame, self.temporal.loc[frame.index])
            frame = frame.join(temporal)
        score = native_predict(self.model, SPEC, native_inputs(frame, "full"), self.rounds)
        return np.column_stack((1 - score, score))


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze():
    if FITTED.exists():
        raise FileExistsError(FITTED)
    pilot_path = DEST / "summary.json"
    data_path = DEST / "features_AB_temporal.parquet"
    pilot = from_json(pilot_path)
    sources = {
        **pilot["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in (pilot_path, data_path)},
    }
    verify_sources(sources)
    x, y, _ = ab_inputs()
    timing = pd.read_parquet(data_path)
    if not timing.index.equals(x.index) or list(timing.columns) != pilot["features"]:
        raise ValueError("A/B temporal features differ from the pilot")
    selected = max(ROUNDS, key=lambda n: pilot["scores"][f"temporal_{n}"]["balanced"])
    model = estimator("xgb3", ROUNDS[-1]).set_params(n_jobs=2)
    model.fit(native_inputs(x.join(timing), "full"), y)
    with tempfile.TemporaryDirectory(dir=DEST, prefix="p22-fit-") as temporary:
        staging = Path(temporary) / "fitted"
        staging.mkdir()
        joblib.dump(model, staging / "xgboost.joblib")
        save_json(staging / "manifest.json", {
            "device": "cpu", "selected_rounds": selected,
            "selected_on": "A/B balanced AP", "pilot": pilot,
            "source_sha256": sources, "model_sha256": sha(staging / "xgboost.joblib"),
        })
        verify_sources(sources)
        staging.rename(FITTED)
    print(f"Froze temporal XGBoost with {selected} rounds using A/B only", flush=True)


def set_c_temporal(stays):
    expected = from_json(OUT / "preparation_manifest.json")["C_sha256"]
    if sha(ARCHIVE) != expected:
        raise ValueError("Set C archive changed")
    remaining = set(stays)
    accepted = []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            stem = Path(member.name).stem
            if not member.isfile() or not stem.isdigit():
                continue
            record_id = int(stem)
            if record_id not in remaining:
                raise ValueError("An unexpected or duplicate Set C record")
            remaining.remove(record_id)
            raw = archive.extractfile(member)
            if raw is None:
                raise ValueError("Cannot read the Set C record")
            _, records, _ = extract_record(raw.read(), record_id)
            accepted.extend(
                (rid, minute, feature, value) for rid, minute, feature, _, value, quality, _ in records
                if feature in GROUPS and quality == "valid"
            )
    if remaining:
        raise ValueError("Set C is missing archive records")
    if sha(ARCHIVE) != expected:
        raise ValueError("Set C archive changed during feature extraction")
    return time_summary(pd.DataFrame(accepted, columns=[ID, MINUTE, "variable", VALUE]), stays)


def eicu_temporal():
    original_cohort, original_frames = eicu_inputs()
    old_sources = from_json(EICU_SCORES / "summary.json")["data_sha256"]
    if source_hashes(DEMO) != old_sources:
        raise ValueError("eICU raw files differ from P8")

    def augment(events, index):
        return time_summary(events.rename(columns={STAY: ID, "minute": MINUTE, "value": VALUE}), index)

    cohort, features, _, _ = build_features(DEMO, feature_extension=augment)
    pd.testing.assert_frame_equal(cohort, original_cohort, check_dtype=False, check_exact=True)
    for mode, raw in original_frames.items():
        if not features[mode].index.equals(raw.index):
            raise ValueError("The eICU temporal features differ in row order")
        np.testing.assert_allclose(features[mode].loc[:, raw.columns], raw, rtol=1e-12, atol=1e-12, equal_nan=True)
    return (cohort, features), old_sources


def evaluate():
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    if not FITTED.exists():
        freeze()
    manifest_path = FITTED / "manifest.json"
    fitted = from_json(manifest_path)
    model_path = FITTED / "xgboost.joblib"
    if sha(model_path) != fitted["model_sha256"] or fitted["device"] != "cpu":
        raise ValueError("Frozen CPU temporal model changed")
    reference = from_json(REFERENCES / "manifest.json")
    protected = {
        **fitted["source_sha256"], **reference["source_sha256"],
        manifest_path.relative_to(OUT).as_posix(): sha(manifest_path),
        model_path.relative_to(OUT).as_posix(): sha(model_path),
        (REFERENCES / "gcs_eicu_scores.parquet").relative_to(OUT).as_posix(): reference["gcs_scores_sha256"],
    }
    verify_sources(protected)
    prior = from_json(P11 / "summary.json")
    model = joblib.load(model_path)
    raw_cols = pd.read_parquet(AB_FEATURES).columns.tolist()
    frame_c, _, _ = load_cohort()
    time_c = set_c_temporal(frame_c.index)
    expected = fitted["pilot"]["features"]
    if list(time_c.columns) != expected:
        raise ValueError("Set C temporal columns differ from A/B")
    print("Evaluating the frozen temporal model with P9 Set C masks", flush=True)
    c_model = TemporalPredictor(model, fitted["selected_rounds"], frame_c, time_c)
    set_c, pairs, intervals = compare_c({TAG: (c_model, raw_cols)}, benchmarks(), reference, prior)
    eicu, eicu_sources = eicu_temporal()
    eicu_model = TemporalPredictor(model, fitted["selected_rounds"])
    rows = compare_eicu({TAG: (eicu_model, raw_cols + expected)}, reference, prior, inputs=eicu)
    if source_hashes(DEMO) != eicu_sources:
        raise ValueError("eICU raw files changed during evaluation")
    verify_sources(protected)
    with tempfile.TemporaryDirectory(dir=DEST, prefix="p22-eval-") as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        pairs.to_csv(staging / "set_c_paired_replicates.csv", index=False)
        save_json(staging / "summary.json", {
            "fitted": fitted, "set_c": set_c, "set_c_clean_ci": intervals, "c_bootstraps": C_BOOTSTRAPS,
            "eicu": rows, "source_sha256": protected, "eicu_raw_sha256": eicu_sources,
        })
        staging.rename(EVALUATION)
    print(f"P22 temporal evaluation saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    evaluate()
