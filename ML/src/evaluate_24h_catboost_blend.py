"""Choose a 24-hour CatBoost blend on P12 A/B, then score it once on holdouts."""

import argparse
import os
import tempfile
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

import evaluate_24h_landmark as landmark
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import contrast_c
from train_gcs_routed_temporal import build_views, matrix, prediction, score, train_one
from train_predictive_boosting import AB_FEATURES, FOLD_IDS, SHIFT_SEED, from_json, verify_sources

OUTPUT = landmark.OUT / "p52_24h_catboost_blend"
P49 = landmark.ROOT / "manifest.json"
CAT = "temporal_cat"
TOLERANCE = 0.005
MIN_GAIN = 0.001


def blend(reference, catboost, has_gcs, recipe):
    weight = recipe["weight"] * (has_gcs if recipe["observed_only"] else 1.0)
    return reference + weight * (catboost - reference)


def score_fold(x, timing, y, fold, held, anchor):
    train_x, train_timing = x.loc[~held], timing.loc[~held]
    seed = SHIFT_SEED + 100 * fold
    primary = anchor["primary"]
    specialist_weight = anchor["blind_weight"]
    if primary == landmark.SUMMARIES or specialist_weight <= 0:
        raise ValueError("Expected P49's temporal model with a GCS-absent specialist")
    models = {
        tag: train_one(tag, train_x, train_timing, y[~held], seed)
        for tag in (primary, landmark.BLIND, CAT)
    }
    original = x.loc[held]
    for view, altered in build_views(original, fold).items():
        values = matrix(original, altered, timing.loc[held])
        has_gcs = altered["GCS__count"].gt(0).to_numpy()
        reference = landmark.use_route(
            prediction(primary, models[primary], values),
            prediction(landmark.BLIND, models[landmark.BLIND], values),
            ~has_gcs,
            specialist_weight,
        )
        yield view, reference, prediction(CAT, models[CAT], values), has_gcs


def fit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_digest = landmark.sha(P49)
    reference_model = from_json(P49)
    selected = reference_model["selected"]
    x, timing, y, folds, protected = landmark.development_24h()
    views = landmark.VIEWS
    reference = {view: np.full(len(y), np.nan) for view in views}
    catboost = {view: np.full(len(y), np.nan) for view in views}
    has_gcs = {view: np.zeros(len(y), dtype=bool) for view in views}
    for fold in FOLD_IDS:
        start = time.perf_counter()
        held = folds == fold
        for view, base, candidate, observed in score_fold(x, timing, y, fold, held, reference_model["recipes"][selected]):
            reference[view][held], catboost[view][held], has_gcs[view][held] = base, candidate, observed
        print(f"P52 A/B fold {fold + 1}: {time.perf_counter() - start:.1f}s", flush=True)

    baseline = score(y, folds, reference)
    expected = reference_model["ab_metrics"][selected]
    np.testing.assert_allclose(list(baseline.values()), [expected[key] for key in baseline], rtol=0, atol=5e-7)
    recipes = {
        f"{scope}_{weight:.2f}": {"weight": weight, "observed_only": scope == "gcs_observed"}
        for scope, weight in product(("all", "gcs_observed"), (0.25, 0.50))
    }
    blend_metrics = {
        name: score(y, folds, {view: blend(reference[view], catboost[view], has_gcs[view], recipe) for view in views})
        for name, recipe in recipes.items()
    }
    eligible = [
        name for name, metrics in blend_metrics.items()
        if metrics["objective"] >= baseline["objective"] + MIN_GAIN
        and all(metrics[key] >= baseline[key] - TOLERANCE for key in ("clean", "full_gcs"))
    ]
    winner = max(eligible, key=lambda name: (blend_metrics[name]["objective"], blend_metrics[name]["clean"]), default=None)
    verify_sources(protected)
    if landmark.sha(P49) != source_digest:
        raise ValueError("P49 reference changed during A/B training")
    with tempfile.TemporaryDirectory(dir=landmark.OUT, prefix="p52-fit-") as temporary:
        staged = Path(temporary) / OUTPUT.name
        staged.mkdir()
        model_hash = None
        if winner:
            final = train_one(CAT, x, timing, y, SHIFT_SEED + 400)
            joblib.dump(final, staged / "catboost.joblib")
            model_hash = landmark.sha(staged / "catboost.joblib")
        landmark.save_json(staged / "manifest.json", {
            "device": "cpu", "ab_stays": len(y), "ab_deaths": int(y.sum()), "ab_fits": len(FOLD_IDS) * 3,
            "final_fits": int(winner is not None), "baseline": baseline, "metrics": blend_metrics,
            "recipes": recipes, "eligible": eligible, "selected": winner, "source_sha256": protected,
            "p49_manifest_sha256": source_digest, "model_sha256": model_hash,
        })
        staged.rename(OUTPUT)
    print(f"P52 selected on A/B: {winner or 'retain P49'}", flush=True)


def scores(models, anchor, candidate, recipe, original, altered, timing):
    baseline = landmark.model_risks(models, anchor, original, altered, timing)["selected"]
    new = prediction(CAT, candidate, matrix(original, altered, timing))
    observed = altered["GCS__count"].gt(0).to_numpy()
    return {"p49": baseline, "p52": blend(baseline, new, observed, recipe)}, observed


def stratified(y, risks, observed):
    results = {}
    for name, mask in (("gcs_observed", observed), ("gcs_missing", ~observed)):
        target = y[mask]
        results[name] = {"stays": int(mask.sum()), "deaths": int(target.sum())}
        if 0 < target.sum() < len(target):
            aps = {model: float(average_precision_score(target, values[mask])) for model, values in risks.items()}
            results[name].update({"ap": aps, "ap_delta": aps["p52"] - aps["p49"]})
    return results


def set_c(models, anchor, candidate, recipe):
    base, timing, target, views, _, meta = landmark.set_c_24h()
    results = {}
    for view, altered in views.items():
        risk, observed = scores(models, anchor, candidate, recipe, base, altered, timing)
        results[view] = {
            "metrics": {tag: landmark.score_metrics(target, values) for tag, values in risk.items()},
            "paired_ci": contrast_c(target, {"p49": risk["p49"]}, {"p52": risk["p52"]}),
            "gcs_strata": stratified(target, risk, observed),
        }
    return {**meta, "views": results}


def demo(models, anchor, candidate, recipe):
    hashes = landmark.source_hashes(landmark.DEMO)
    if hashes != from_json(landmark.P8 / "summary.json")["data_sha256"]:
        raise ValueError("Unexpected eICU Demo tables")
    cohort, inputs, _, _, _ = landmark.demo_inputs_24h(pd.read_parquet(AB_FEATURES).columns)
    modes = {}
    strata = {}
    for mode, (base, timing) in inputs.items():
        risk, observed = scores(models, anchor, candidate, recipe, base, base, timing)
        modes[mode] = risk
        strata[mode] = stratified(cohort.y.to_numpy(), risk, observed)
    apache, included, counts = landmark.demo_apache(cohort)
    matched = cohort.loc[included]
    matched_risks = {
        mode: {"apache_iva": apache, **{name: values[included] for name, values in risks.items()}}
        for mode, risks in modes.items()
    }
    matched_metrics = {
        mode: {name: landmark.score_metrics(matched.y.to_numpy(), values) for name, values in risks.items()}
        for mode, risks in matched_risks.items()
    }
    all_metrics = {
        mode: {name: landmark.score_metrics(cohort.y.to_numpy(), values) for name, values in risks.items()}
        for mode, risks in modes.items()
    }
    if landmark.source_hashes(landmark.DEMO) != hashes or landmark.sha(landmark.APACHE_FILE) != landmark.APACHE_SHA256:
        raise ValueError("eICU Demo sources changed during evaluation")
    return {
        "cohort": counts, "all_metrics": all_metrics, "matched_metrics": matched_metrics,
        "full_paired_p49": paired_intervals(
            cohort, {mode: risks["p49"] for mode, risks in modes.items()}, {mode: risks["p52"] for mode, risks in modes.items()},
        ),
        "matched_paired_apache": paired_intervals(
            matched, dict.fromkeys(matched_risks, apache), {mode: risks["p52"] for mode, risks in matched_risks.items()},
        ),
        "gcs_strata": strata,
    }


def evaluate():
    path = OUTPUT / "evaluation.json"
    if path.exists():
        raise FileExistsError(path)
    frozen = from_json(OUTPUT / "manifest.json")
    if not frozen["selected"]:
        raise ValueError("A/B retained P49. No new candidate to evaluate.")
    model = OUTPUT / "catboost.joblib"
    if landmark.sha(P49) != frozen["p49_manifest_sha256"] or landmark.sha(model) != frozen["model_sha256"]:
        raise ValueError("A frozen model source changed")
    verify_sources(frozen["source_sha256"])
    models, anchor = landmark.frozen()
    recipe = frozen["recipes"][frozen["selected"]]
    candidate = joblib.load(model)
    result = {
        "set_c": set_c(models, anchor, candidate, recipe),
        "eicu_demo": demo(models, anchor, candidate, recipe),
    }
    staged = path.with_suffix(".tmp")
    landmark.save_json(staged, result)
    staged.rename(path)
    print(f"Saved CPU evaluation to {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with threadpool_limits(limits=2):
        {"fit": fit, "evaluate": evaluate}[args.phase]()
