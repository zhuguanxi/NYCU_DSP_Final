"""Select a model without explicit monitoring signals on P12 A/B."""

import argparse
import tempfile
from pathlib import Path

from select_robust_ensemble import (
    OBJECTIVE, OUTPUT as P28, P26, TOLERANCE, VIEWS, blend, build_views, folded_elastic, score,
)

import joblib
import numpy as np
import pandas as pd

from evaluate_eicu_confirmation import frozen_ensemble, frozen_models
from evaluate_gcs_routed_temporal import (
    EVALUATION as P26_EVAL, P22_EVAL, P23, P24, RoutedPredictor, checked_metrics, compare_eicu,
)
from evaluate_ma_gbt_p12 import P11, REFERENCES
from evaluate_predictive_boosting import benchmarks, compare_c
from evaluate_robust_ensemble import EVALUATION as P28_EVAL, Ensemble
from evaluate_temporal_features import save_json, set_c_temporal
from pilot_temporal_features import DEST as P22, temporal_view
from prepare import OUT, sha
from stress_missingness_c import METRICS, load_cohort
from train_gcs_routed_temporal import mapped_shift
from train_predictive_boosting import (
    AB_FEATURES, AB_SELECTION, FOLD_IDS, ROUNDS, SHIFT_SEED, ab_inputs, estimator, from_json,
    native_fit, native_inputs, native_predict, verify_sources,
)

DEST = OUT / "p31_monitoring_ablation"
EVALUATION = DEST / "evaluation"
ANCHOR = "p28"
NEW = "monitoring_ablation"
POLICIES = {
    "xgb_plain": {"algorithm": "xgb3", "augment": False},
    "xgb_augmented": {"algorithm": "xgb3", "augment": True},
    "lgb_augmented": {"algorithm": "lgb15", "augment": True},
}
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
MONITORING = {"count", "missing", "slope_defined", "std_defined", "hours_since_last", "covered_hours"}


def usable(columns):
    return [column for column in columns if column.split("__")[-1] not in MONITORING]


def matrix(original, frame, timing, columns):
    combined = frame.join(temporal_view(original, frame, timing))
    return native_inputs(combined.loc[:, columns], "full")


def score_views(y, folds, prediction):
    result = score(y, folds, prediction)
    result["objective"] = sum(weight * result[key] for key, weight in OBJECTIVE.items())
    return result


def previous_scores(x, y, folds, selection, routed):
    folder = OUT / "runs" / from_json(AB_SELECTION)["XGBoost_Elastic"]
    elastic, absent = folded_elastic(x, y, folds, folder)
    settings = routed["selected"]
    primary, fallback, mass = (settings[key] for key in ("primary", "fallback", "weight"))
    rule = selection["rule"]
    result = {}
    with np.load(P26 / "ab_candidate_predictions.npz", allow_pickle=False) as stored:
        for view in VIEWS:
            lead = stored[f"{primary}_{view}"]
            other = stored[f"{fallback}_{view}"]
            reference = np.where(absent[view], (1 - mass) * lead + mass * other, lead)
            result[view] = reference if rule is None else blend(
                reference, elastic[rule["elastic"]][view], absent[view], rule,
            )
    original = selection["a_b_metrics"][selection["winner"]]
    np.testing.assert_allclose(
        list(score_views(y, folds, result).values()), list(original.values()), atol=1e-12, rtol=0,
    )
    return result


def select():
    if DEST.exists():
        raise FileExistsError(DEST)
    selection_path, routed_path = P28 / "manifest.json", P26 / "manifest.json"
    selection, routed = from_json(selection_path), from_json(routed_path)
    timing_path = P22 / "features_AB_temporal.parquet"
    sources = {
        **selection["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in (selection_path, routed_path, timing_path)},
    }
    verify_sources(sources)

    x, y, folds = ab_inputs()
    temporal = pd.read_parquet(timing_path)
    if not x.index.equals(temporal.index):
        raise ValueError("A/B timing features do not align")
    columns = usable([*x.columns, *temporal.columns])
    predictions = {tag: {view: np.full(len(y), np.nan) for view in VIEWS} for tag in POLICIES}
    reference = previous_scores(x, y, folds, selection, routed)
    for fold in FOLD_IDS:
        train = folds != fold
        early, held = x.loc[train], x.loc[~train]
        train_time, held_time = temporal.loc[train], temporal.loc[~train]
        base = matrix(early, early, train_time, columns)
        shifted = matrix(early, mapped_shift(early, SHIFT_SEED + 100 * fold), train_time, columns)
        held_views = {view: matrix(held, frame, held_time, columns) for view, frame in build_views(held, fold).items()}
        for tag, spec in POLICIES.items():
            model = native_fit(spec, base, y[train], shifted, ROUNDS[-1])
            for view, values in held_views.items():
                predictions[tag][view][~train] = native_predict(model, spec, values, ROUNDS[-1])
            print(f"Completed monitoring ablation A/B fold {fold}: {tag}", flush=True)

    options = {f"{tag}:{weight}": (tag, weight) for tag in POLICIES for weight in WEIGHTS}
    scores = {ANCHOR: score_views(y, folds, reference)}
    for name, (tag, weight) in options.items():
        scores[name] = score_views(y, folds, {
            view: (1 - weight) * reference[view] + weight * predictions[tag][view] for view in VIEWS
        })
    eligible = [
        key for key, value in scores.items()
        if all(value[metric] >= scores[ANCHOR][metric] - tol for metric, tol in TOLERANCE.items())
    ]
    winner = max(eligible, key=lambda key: (scores[key]["objective"], scores[key]["clean"]))
    choice = options.get(winner)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p31-ablation-") as temporary:
        stage = Path(temporary) / DEST.name
        stage.mkdir()
        digest = None
        if choice:
            tag, _ = choice
            base = matrix(x, x, temporal, columns)
            shifted = matrix(x, mapped_shift(x, SHIFT_SEED + 400), temporal, columns)
            model = native_fit(POLICIES[tag], base, y, shifted, ROUNDS[-1])
            joblib.dump(model, stage / "model.joblib")
            digest = sha(stage / "model.joblib")
        verify_sources(sources)
        save_json(stage / "manifest.json", {
            "dataset": "P12 A/B", "device": "cpu", "winner": winner, "selected": choice,
            "a_b_metrics": scores, "eligible": eligible, "columns": columns, "model_sha256": digest,
            "before_columns": x.shape[1] + temporal.shape[1], "source_sha256": sources,
            "a_b_fits": len(FOLD_IDS) * (1 + len(POLICIES)), "final_fits": int(choice is not None),
        })
        stage.rename(DEST)
    print(f"P31 A/B selection: {winner}; {len(eligible)} feasible", flush=True)


class AblationPredictor:
    def __init__(self, anchor, model, spec, weight, columns, original=None, timing=None):
        self.anchor, self.model, self.spec, self.weight = anchor, model, spec, weight
        self.columns, self.original, self.timing = columns, original, timing

    def predict_proba(self, frame):
        if self.timing is None:
            values = native_inputs(frame.loc[:, self.columns], "full")
        else:
            values = matrix(self.original.loc[frame.index], frame, self.timing.loc[frame.index], self.columns)
        ablated = native_predict(self.model, self.spec, values, ROUNDS[-1])
        combined = self.weight * ablated
        if self.weight < 1:
            combined += (1 - self.weight) * self.anchor.predict_proba(frame)[:, 1]
        return np.column_stack((1 - combined, combined))


def evaluate():
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    manifest_path = DEST / "manifest.json"
    fit = from_json(manifest_path)
    if fit["selected"] is None:
        print("A/B retained the already evaluated P28 model; no additional holdout scoring.", flush=True)
        return
    tag, weight = fit["selected"]
    model_path = DEST / "model.joblib"
    paths = [manifest_path, *(parent / "summary.json" for parent in (P22_EVAL, P23, P24, P26_EVAL, P28_EVAL, P11)), REFERENCES / "manifest.json"]
    p22, p23, p24, p26, p28, p11, ref = map(from_json, paths[1:])
    routed, temporal, gcs, extras, protected = frozen_models()
    rule, _, _ = frozen_ensemble(protected)
    protected.update(fit["source_sha256"])
    protected.update(p28["source_sha256"])
    protected.update({path.relative_to(OUT).as_posix(): sha(path) for path in paths})
    protected[model_path.relative_to(OUT).as_posix()] = fit["model_sha256"]
    verify_sources(protected)
    model = joblib.load(model_path)
    original, y, _ = load_cohort()
    timing = set_c_temporal(original.index)
    if list(timing.columns) != extras:
        raise ValueError("Set C temporal schema changed")
    columns = list(original.columns)
    routed_c = RoutedPredictor(routed.rule, routed.models, original, timing)
    p28_c = Ensemble(routed_c, gcs, rule)
    p28_c_rows = {row["scenario"]: row for row in p28["set_c"]}
    checked_metrics(y, p28_c.predict_proba(original)[:, 1], p28_c_rows["clean"])
    finalist_c = AblationPredictor(p28_c, model, POLICIES[tag], weight, fit["columns"], original, timing)
    c, masks, c_ci = compare_c(
        {NEW: (finalist_c, columns)}, {**benchmarks(), ANCHOR: (p28_c, columns)}, ref, p11,
    )
    for row in c:
        for metric in METRICS:
            row[f"{metric}_p28"] = p28_c_rows[row["scenario"]][metric]
            row[f"delta_{metric}_vs_p28"] = row[metric] - row[f"{metric}_p28"]
    p28_external = Ensemble(routed, gcs, rule)
    finalist_ext = AblationPredictor(p28_external, model, POLICIES[tag], weight, fit["columns"])
    eicu, raw_sources = compare_eicu(
        finalist_ext, temporal, columns, extras, p22, p23, p24, additional={ANCHOR: p28_external, "routed": routed},
    )
    if raw_sources != p26["eicu_raw_sha256"] or raw_sources != p28["eicu_raw_sha256"]:
        raise ValueError("Historical eICU inputs differ")
    p28_rows = {(row["alignment"], row["mode"]): row["metrics"] for row in p28["eicu"]}
    for row in eicu:
        replayed = row["comparison"][ANCHOR]["metrics"]
        expected = p28_rows[row["alignment"], row["mode"]]
        np.testing.assert_allclose([replayed[m] for m in METRICS], [expected[m] for m in METRICS], atol=1e-12, rtol=0)
    verify_sources(protected)
    with tempfile.TemporaryDirectory(dir=DEST, prefix="p31-evaluation-") as temporary:
        stage = Path(temporary) / "evaluation"
        stage.mkdir()
        masks.to_csv(stage / "set_c_paired_replicates.csv", index=False)
        save_json(stage / "summary.json", {
            "training": fit, "set_c": c, "set_c_clean_ci": c_ci, "eicu": eicu,
            "source_sha256": protected, "eicu_raw_sha256": raw_sources,
        })
        stage.rename(EVALUATION)
    print(f"P31 frozen evaluation saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "evaluate"))
    {"select": select, "evaluate": evaluate}[parser.parse_args().stage]()
