"""Evaluate one P28 selection on the original frozen Set C and eICU Demo."""

import tempfile
from pathlib import Path

from select_robust_ensemble import ANCHOR, OUTPUT, blend

import joblib
import numpy as np
import pandas as pd

from evaluate_eicu_confirmation import frozen_models
from evaluate_gcs_routed_temporal import (
    EVALUATION as P26_EVAL, P22_EVAL, P23, P24, RoutedPredictor, checked_metrics, compare_eicu,
)
from evaluate_ma_gbt_p12 import P11, REFERENCES
from evaluate_predictive_boosting import benchmarks, compare_c, no_category_warning
from evaluate_temporal_features import save_json, set_c_temporal
from prepare import OUT, sha
from stress_missingness_c import METRICS, load_cohort
from train_missingness_augmentation import FROZEN, from_json
from train_predictive_boosting import AB_FEATURES, verify_sources

EVALUATION = OUTPUT / "evaluation"
NEW = "robust_ensemble"


class Ensemble:
    def __init__(self, routed, elastic, rule):
        self.routed = routed
        self.elastic = elastic
        self.rule = rule

    def predict_proba(self, frame):
        routed = self.routed.predict_proba(frame)
        if self.rule is None:
            return routed
        pipe, columns = self.elastic
        other = no_category_warning(pipe, frame.loc[:, columns])
        score = blend(routed[:, 1], other, frame["GCS__count"].eq(0).to_numpy(), self.rule)
        return np.column_stack((1 - score, score))


def evaluate():
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    fit_path = OUTPUT / "manifest.json"
    fit = from_json(fit_path)
    if (fit["device"], fit["dataset"], fit["final_fits"]) != ("cpu", "P12 A/B", 0):
        raise ValueError("Expected the P28 A/B frozen mixture")
    routed, temporal, gcs, extras, sources = frozen_models()
    paths = [fit_path, *(parent / "summary.json" for parent in (P22_EVAL, P23, P24, P26_EVAL, P11)), REFERENCES / "manifest.json"]
    p22, p23, p24, p26, p11, reference = map(from_json, paths[1:])
    protected = {
        **sources, **p26["source_sha256"], **fit["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in paths},
    }
    verify_sources(protected)
    pipe, _, reduced_columns = joblib.load(FROZEN / "Elastic.joblib")
    pipe.named_steps["model"].set_params(device="cpu", n_jobs=2)
    elastic = {"frozen_elastic": (pipe, reduced_columns), "gcs_only_elastic": gcs}
    rule = fit["rule"]
    selected = elastic[rule["elastic"]] if rule is not None else None
    full_columns = pd.read_parquet(AB_FEATURES).columns.tolist()

    raw, y, _ = load_cohort()
    timing = set_c_temporal(raw.index)
    if list(timing.columns) != extras or not timing.index.equals(raw.index):
        raise ValueError("Set C temporal schema changed")
    routed_c = RoutedPredictor(routed.rule, routed.models, raw, timing)
    c_prior = {row["scenario"]: row for row in p26["set_c"]}
    checked_metrics(y, routed_c.predict_proba(raw.loc[:, full_columns])[:, 1], c_prior["clean"])
    print("P28 Set C frozen score and missingness replay", flush=True)
    set_c, masks, c_ci = compare_c(
        {NEW: (Ensemble(routed_c, selected, rule), full_columns)},
        {**benchmarks(), ANCHOR: (routed_c, full_columns)}, reference, p11,
    )
    for row in set_c:
        for metric in METRICS:
            row[f"{metric}_routed"] = c_prior[row["scenario"]][metric]
            row[f"delta_{metric}_vs_routed"] = row[metric] - row[f"{metric}_routed"]

    print("P28 eICU frozen score, both alignments", flush=True)
    eicu, raw_sources = compare_eicu(
        Ensemble(routed, selected, rule), temporal, full_columns, extras, p22, p23, p24,
        additional={ANCHOR: routed},
    )
    old_eicu = {(row["alignment"], row["mode"]): row["metrics"] for row in p26["eicu"]}
    for row in eicu:
        expected = old_eicu[row["alignment"], row["mode"]]
        actual = row["comparison"][ANCHOR]["metrics"]
        np.testing.assert_allclose([actual[m] for m in METRICS], [expected[m] for m in METRICS], rtol=0, atol=1e-12)
    if raw_sources != p26["eicu_raw_sha256"]:
        raise ValueError("Raw eICU inputs changed since P26")
    verify_sources(protected)
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="p28-evaluation-") as temporary:
        stage = Path(temporary) / "evaluation"
        stage.mkdir()
        masks.to_csv(stage / "set_c_paired_replicates.csv", index=False)
        save_json(stage / "summary.json", {
            "training": fit, "set_c": set_c, "set_c_clean_ci": c_ci, "eicu": eicu,
            "source_sha256": protected, "eicu_raw_sha256": raw_sources,
        })
        stage.rename(EVALUATION)
    print(f"Frozen ensemble evaluation saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    evaluate()
