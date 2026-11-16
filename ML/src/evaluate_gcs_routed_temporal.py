"""Score one A/B-frozen GCS routing rule on Set C and both eICU alignments."""

import tempfile
from pathlib import Path

from train_gcs_routed_temporal import BASELINE, OUTPUT, prediction

import joblib
import numpy as np
import pandas as pd

from eicu_demo_features import DEMO, STAY
from evaluate_eicu_demo import MODES, source_hashes
from evaluate_eicu_urine_mapping import OUTPUT as P23, urine_events, urine_values
from evaluate_ma_gbt_p12 import P11, REFERENCES
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import benchmarks, compare_c, no_category_warning
from evaluate_temporal_features import (
    EVALUATION as P22_EVAL, FITTED as P22_FITTED, TemporalPredictor, eicu_temporal, save_json, set_c_temporal,
)
from evaluate_temporal_urine_alignment import OUTPUT as P24, mapped_elastic_controls
from pilot_temporal_features import ID, MINUTE, VALUE, temporal_view, time_summary
from prepare import OUT, sha
from stress_missingness_c import METRICS, load_cohort, score_metrics
from train_missingness_augmentation import from_json
from train_predictive_boosting import AB_FEATURES, native_inputs, verify_sources

EVALUATION = OUTPUT / "evaluation"
NEW = "gcs_routed_temporal"


class RoutedPredictor:
    def __init__(self, components, models, clean=None, temporal=None):
        self.rule, self.models = components, models
        self.clean, self.temporal = clean, temporal

    def predict_proba(self, frame):
        inputs = frame if self.temporal is None else frame.join(
            temporal_view(self.clean.loc[frame.index], frame, self.temporal.loc[frame.index])
        )
        values = native_inputs(inputs, "full")
        primary = self.rule["primary"]
        scores = prediction(primary, self.models[primary], values)
        is_missing = frame["GCS__count"].eq(0).to_numpy()
        weight, fallback = self.rule["weight"], self.rule["fallback"]
        if weight and is_missing.any():
            if fallback == BASELINE:
                pipeline, cols = self.models[BASELINE]
                alternate = no_category_warning(pipeline, frame.loc[is_missing, cols])
            else:
                alternate = prediction(fallback, self.models[fallback], values[is_missing])
            scores[is_missing] = (1 - weight) * scores[is_missing] + weight * alternate
        return np.column_stack((1 - scores, scores))


def checked_metrics(y, probabilities, prior):
    measured = score_metrics(y, probabilities)
    np.testing.assert_allclose([measured[key] for key in METRICS], [prior[key] for key in METRICS], rtol=0, atol=1e-12)


def compare_eicu(model, ref_model, raw_cols, extras, previous, p23, p24, *, additional=None):
    (cohort, frames), raw_sha = eicu_temporal()
    y = cohort.y.to_numpy()
    if raw_sha != p23["raw_sha256"] or raw_sha != p24["eicu_raw_sha256"]:
        raise ValueError("eICU raw sources differ across old comparisons")
    events, coverage = urine_events(cohort.index)
    if coverage != p23["coverage"]:
        raise ValueError("The original urine mapping changed")
    urine = urine_values(cohort.index, events)
    timing = time_summary(events.rename(columns={STAY: ID, "minute": MINUTE, "value": VALUE}), cohort.index)
    timing = timing.filter(like="temporal_Urine__")
    mapped = {}
    for mode, view in frames.items():
        updated = view.copy()
        updated.loc[:, urine.columns] = urine
        updated.loc[:, timing.columns] = timing
        mapped[mode] = updated
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    expected_gcs = from_json(REFERENCES / "manifest.json")["gcs_scores_sha256"]
    if sha(gcs_path) != expected_gcs:
        raise ValueError("Frozen GCS-only reference changed")
    old_gcs = pd.read_parquet(gcs_path)
    if not old_gcs.index.equals(cohort.index):
        raise ValueError("GCS-only reference has different stays")
    p22_by_mode = {row["mode"]: row for row in previous["eicu"]}
    p24_by_mode = {row["mode"]: row for row in p24["rows"]}
    rows = []
    full_cols = raw_cols + extras
    for alignment, inputs in (("original", frames), ("urine_mapped", mapped)):
        reference_temporal = {mode: ref_model.predict_proba(inputs[mode].loc[:, full_cols])[:, 1] for mode in MODES}
        controls = {"temporal": reference_temporal}
        if additional:
            controls.update({
                tag: {mode: pipe.predict_proba(inputs[mode].loc[:, full_cols])[:, 1] for mode in MODES}
                for tag, pipe in additional.items()
            })
        if alignment == "urine_mapped":
            controls.update(mapped_elastic_controls({mode: f.loc[:, raw_cols] for mode, f in mapped.items()}, y, p23))
        else:
            controls["gcs_elastic_original"] = {mode: old_gcs[f"{mode}_Elastic"].to_numpy() for mode in MODES}
        for mode, scores in reference_temporal.items():
            expect = p24_by_mode[mode]["temporal_mapped"] if alignment == "urine_mapped" else p22_by_mode[mode]
            checked_metrics(y, scores, expect)
        probs = {mode: model.predict_proba(inputs[mode].loc[:, full_cols])[:, 1] for mode in MODES}
        intervals = {tag: paired_intervals(cohort, baseline, probs) for tag, baseline in controls.items()}
        for mode, predicted in probs.items():
            measures = score_metrics(y, predicted)
            result = {
                "alignment": alignment, "mode": mode, "metrics": measures,
                "gcs_absent": int(inputs[mode]["GCS__count"].eq(0).sum()), "comparison": {},
            }
            for tag, baseline in controls.items():
                base = score_metrics(y, baseline[mode])
                result["comparison"][tag] = {
                    "metrics": base, "delta": {m: measures[m] - value for m, value in base.items()},
                    "paired_ci": intervals[tag][mode],
                }
            rows.append(result)
    if source_hashes(DEMO) != raw_sha:
        raise ValueError("External raw sources changed during the evaluation")
    return rows, raw_sha


def evaluate():
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    fit_path = OUTPUT / "manifest.json"
    fit = from_json(fit_path)
    if (fit["dataset"], fit["device"]) != ("P12 A/B", "cpu") or fit["routing"] != "use fallback weight when GCS__count == 0":
        raise ValueError("Expected the A/B-frozen CPU routing rule")
    p22_path, p23_path, p24_path = P22_EVAL / "summary.json", P23 / "summary.json", P24 / "summary.json"
    p22, p23, p24 = [from_json(path) for path in (p22_path, p23_path, p24_path)]
    p22_model = P22_FITTED / "xgboost.joblib"
    if sha(p22_model) != p22["fitted"]["model_sha256"]:
        raise ValueError("The frozen P22 comparator changed")
    ref_path = REFERENCES / "manifest.json"
    reference = from_json(ref_path)
    protected = {
        **fit["source_sha256"], **p22["source_sha256"], **reference["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in (p22_path, p23_path, p24_path, fit_path, ref_path)},
        **{(OUTPUT / f"{tag}.joblib").relative_to(OUT).as_posix(): digest for tag, digest in fit["model_sha256"].items()},
        (REFERENCES / "gcs_eicu_scores.parquet").relative_to(OUT).as_posix(): reference["gcs_scores_sha256"],
    }
    verify_sources(protected)
    controls = benchmarks()
    models = {BASELINE: controls[BASELINE], **{tag: joblib.load(OUTPUT / f"{tag}.joblib") for tag in fit["model_sha256"]}}
    cols = pd.read_parquet(AB_FEATURES).columns.tolist()
    extras = p22["fitted"]["pilot"]["features"]
    frame, y, _ = load_cohort()
    temporal = set_c_temporal(frame.index)
    if list(temporal.columns) != extras:
        raise ValueError("Set C temporal schema differs")
    routed_c = RoutedPredictor(fit["selected"], models, frame, temporal)
    p22_model = joblib.load(p22_model)
    rounds = p22["fitted"]["selected_rounds"]
    reference_c = TemporalPredictor(p22_model, rounds, frame, temporal)
    base_c = {row["scenario"]: row for row in p22["set_c"]}
    checked_metrics(y, reference_c.predict_proba(frame.loc[:, cols])[:, 1], base_c["clean"])
    prior = from_json(P11 / "summary.json")
    print("Set C frozen evaluation of the routed pipeline", flush=True)
    set_c, masks, c_ci = compare_c(
        {NEW: (routed_c, cols)}, {**controls, "temporal": (reference_c, cols)}, reference, prior
    )
    for row in set_c:
        prev = base_c[row["scenario"]]
        for metric in METRICS:
            row[f"{metric}_temporal"] = prev[metric]
            row[f"delta_{metric}_vs_temporal"] = row[metric] - prev[metric]
    model = RoutedPredictor(fit["selected"], models)
    eicu, raw_sha = compare_eicu(model, TemporalPredictor(p22_model, rounds), cols, extras, p22, p23, p24)
    verify_sources(protected)
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="routed-eval-") as temporary:
        stage = Path(temporary) / "evaluation"
        stage.mkdir()
        masks.to_csv(stage / "set_c_paired_replicates.csv", index=False)
        save_json(stage / "summary.json", {
            "training": fit, "set_c": set_c, "set_c_clean_ci": c_ci, "eicu": eicu,
            "set_c_gcs_absent": int(frame["GCS__count"].eq(0).sum()),
            "source_sha256": protected, "eicu_raw_sha256": raw_sha,
        })
        stage.rename(EVALUATION)
    print(f"Routed pipeline evaluation saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    evaluate()
