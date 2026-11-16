"""Select an Elastic and GCS-routed mixture using only original P12 A/B folds."""

import itertools
import tempfile
from pathlib import Path

from train_gcs_routed_temporal import OUTPUT as P26, VIEWS, build_views, score
from train_predictive_boosting import (
    AB_FEATURES, AB_SELECTION, FOLD_IDS, FROZEN, GCS_ONLY_OUTPUT, GCS_SEED,
    ab_inputs, fit_augmented, from_json, mask_observed, validate_oof,
    verified_template, verify_sources,
)

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from evaluate_predictive_boosting import no_category_warning
from evaluate_temporal_features import save_json
from prepare import OUT, sha

OUTPUT = OUT / "p28_robust_ensemble"
ANCHOR = "routed"
ELASTIC = ("frozen_elastic", "gcs_only_elastic")
WEIGHTS = (0.0, 0.25, 0.5, 0.75)
TOLERANCE = {"clean": 0.003, "full_gcs": 0.005}
OBJECTIVE = {"clean": 0.35, "shift": 0.15, "mapped": 0.35, "full_gcs": 0.15}


def blend(anchor, elastic, absent, rule):
    weight = np.where(absent, rule["absent"], rule["observed"])
    if rule["scale"] == "probability":
        return (1 - weight) * anchor + weight * elastic
    a = logit(np.clip(anchor, 1e-7, 1 - 1e-7))
    b = logit(np.clip(elastic, 1e-7, 1 - 1e-7))
    return np.where(weight == 0, anchor, expit((1 - weight) * a + weight * b))


def folded_elastic(x, y, folds, folder):
    predictions = {tag: {view: np.full(len(y), np.nan) for view in VIEWS} for tag in ELASTIC}
    absent = {view: np.zeros(len(y), dtype=bool) for view in VIEWS}
    scores = pd.read_csv(folder / "oof.csv").set_index("RecordID")
    feature_digest = sha(AB_FEATURES)
    for fold in FOLD_IDS:
        train = folds != fold
        early, held = x.loc[train], x.loc[~train]
        template, columns = verified_template(folder / f"fold{fold}.joblib", feature_digest)
        validate_oof(template, columns, held, y[~train], fold, scores)
        masked, changed, _ = mask_observed(early, {"GCS": 60}, GCS_SEED + 100 * fold)
        augmented, elapsed = fit_augmented(template, columns, early, y[train], masked, changed)
        for view, frame in build_views(held, fold).items():
            absent[view][~train] = frame["GCS__count"].eq(0).to_numpy()
            for tag, pipe in zip(ELASTIC, (template, augmented), strict=True):
                predictions[tag][view][~train] = no_category_warning(pipe, frame.loc[:, columns])
        print(f"Reproduced Elastic fold {fold}; GCS-only fit {elapsed:.1f}s", flush=True)
    return predictions, absent


def choose(y, folds, observed, absent):
    rules = {
        f"{tag}:{scale}:{available:.2f}:{missing:.2f}": {
            "elastic": tag, "scale": scale, "observed": available, "absent": missing,
        }
        for tag, scale, available, missing in itertools.product(
            ELASTIC, ("probability", "log_odds"), WEIGHTS, WEIGHTS
        )
        if available or missing
    }
    summary = {}
    for tag, values in observed.items():
        summary[tag] = score(y, folds, values)
    for tag, rule in rules.items():
        values = {
            view: blend(observed[ANCHOR][view], observed[rule["elastic"]][view], absent[view], rule)
            for view in VIEWS
        }
        summary[tag] = score(y, folds, values)
    for metrics in summary.values():
        metrics["objective"] = sum(weight * metrics[key] for key, weight in OBJECTIVE.items())
    guard = {key: summary[ANCHOR][key] for key in TOLERANCE}
    eligible = [
        tag for tag in (ANCHOR, *rules)
        if all(summary[tag][key] >= guard[key] - tolerance for key, tolerance in TOLERANCE.items())
    ]
    winner = max(eligible, key=lambda tag: (summary[tag]["objective"], summary[tag]["clean"]))
    return summary, guard, eligible, winner, rules.get(winner)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    routing_path = P26 / "manifest.json"
    oof_path = P26 / "ab_candidate_predictions.npz"
    routed = from_json(routing_path)
    gcs_path = GCS_ONLY_OUTPUT / "manifest.json"
    gcs = from_json(gcs_path)
    frozen_file, gcs_file = FROZEN / "Elastic.joblib", GCS_ONLY_OUTPUT / "Elastic.joblib"
    if (
        routed["dataset"] != "P12 A/B" or routed["device"] != "cpu"
        or gcs["selected_rates_percent"] != {"GCS": 60}
        or gcs["original_model_sha256"]["Elastic"] != sha(frozen_file)
        or gcs["model_sha256"]["Elastic"] != sha(gcs_file)
    ):
        raise ValueError("Expected frozen P26 and Elastic A/B recipes")
    folder = OUT / "runs" / from_json(AB_SELECTION)["XGBoost_Elastic"]
    paths = [
        routing_path, oof_path, gcs_path, frozen_file, gcs_file, folder / "oof.csv",
        folder / "config.json", *(folder / f"fold{k}.joblib" for k in FOLD_IDS),
    ]
    sources = {
        **routed["source_sha256"], **{p.relative_to(OUT).as_posix(): sha(p) for p in paths},
        **{(P26 / f"{tag}.joblib").relative_to(OUT).as_posix(): digest for tag, digest in routed["model_sha256"].items()},
    }
    verify_sources(sources)
    x, y, folds = ab_inputs()
    if from_json(folder / "config.json")["data_sha"] != sha(AB_FEATURES):
        raise ValueError("Elastic folds do not match A/B feature input")
    observed, absent = folded_elastic(x, y, folds, folder)
    primary, fallback, mass = (routed["selected"][key] for key in ("primary", "fallback", "weight"))
    with np.load(oof_path, allow_pickle=False) as stored:
        observed[ANCHOR] = {
            view: np.where(
                absent[view], (1 - mass) * stored[f"{primary}_{view}"] + mass * stored[f"{fallback}_{view}"],
                stored[f"{primary}_{view}"],
            )
            for view in VIEWS
        }
    if any(
        values.shape != y.shape or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
        for group in observed.values() for values in group.values()
    ):
        raise ValueError("Invalid out-of-fold scores")
    for key, value in score(y, folds, observed[ANCHOR]).items():
        np.testing.assert_allclose(value, routed["a_b_metrics"][routed["winner"]][key], rtol=0, atol=1e-12)
    summary, guard, eligible, winner, rule = choose(y, folds, observed, absent)
    verify_sources(sources)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p28-selection-") as temporary:
        stage = Path(temporary) / OUTPUT.name
        stage.mkdir()
        save_json(stage / "manifest.json", {
            "dataset": "P12 A/B", "device": "cpu", "winner": winner, "rule": rule,
            "a_b_metrics": summary, "objective_weights": OBJECTIVE, "guard_reference": guard,
            "guard_tolerance": TOLERANCE, "eligible": eligible, "a_b_fits": len(FOLD_IDS),
            "final_fits": 0, "source_sha256": sources,
        })
        stage.rename(OUTPUT)
    print(f"Frozen {winner}; {len(eligible)} eligible; {summary[winner]}", flush=True)


if __name__ == "__main__":
    run()
