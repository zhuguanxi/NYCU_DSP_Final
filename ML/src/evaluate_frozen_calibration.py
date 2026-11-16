"""Use only A/B fold scores to calibrate risk and freeze classification thresholds."""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

from audit_p12_outcome_timing import replay_frozen_c
from eicu_demo_features import DEMO
from evaluate_eicu_confirmation import frozen_ensemble, frozen_models
from evaluate_eicu_demo import BOOTSTRAPS, SEED, source_hashes
from evaluate_gcs_routed_temporal import checked_metrics
from evaluate_hourly_sequence import ABLATION, GROUPS, PYTORCH, ROOT as P33, eicu_sequence_cases, infer
from evaluate_predictive_boosting import C_BOOTSTRAPS, C_BOOTSTRAP_SEED, benchmarks, no_category_warning
from evaluate_robust_ensemble import EVALUATION as P28_EVAL, Ensemble
from evaluate_temporal_features import save_json
from prepare import OUT, sha
from sequence_blending import blend
from stress_missingness_c import score_metrics
from train_predictive_boosting import (
    AB_FEATURES, AB_FOLDS, AB_LABELS, AB_SELECTION, FOLD_IDS, ab_inputs, from_json, validate_oof,
    verified_template, verify_sources,
)

ROOT = OUT / "p42_frozen_calibration"
ORIGINALS = {"frozen_full": "XGBoost_Full", "frozen_elastic": "XGBoost_Elastic"}
EPSILON = 1e-7


def log_odds(scores):
    return logit(np.clip(np.asarray(scores, dtype=np.float64), EPSILON, 1 - EPSILON))


def calibrate(scores, recipe):
    return expit(recipe["intercept"] + recipe["slope"] * log_odds(scores))


def binary_metrics(y, score, threshold):
    flagged = score >= threshold
    tp = int(np.count_nonzero(flagged & (y == 1)))
    positive = int(flagged.sum())
    sensitivity = tp / int(y.sum())
    ppv = tp / positive if positive else 0.0
    return {"event_1": min(sensitivity, ppv), "sensitivity": sensitivity, "ppv": ppv, "flagged": positive}


def learn(y, score):
    fit = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(log_odds(score)[:, None], y)
    recipe = {"intercept": float(fit.intercept_[0]), "slope": float(fit.coef_[0, 0])}
    if recipe["slope"] <= 0 or not np.isfinite(list(recipe.values())).all():
        raise ValueError("Expected a finite, increasing calibration map")
    precision, recall, thresholds = precision_recall_curve(y, score)
    event1 = np.minimum(precision[:-1], recall[:-1])
    chosen = float(thresholds[np.flatnonzero(event1 == event1.max())[-1]])
    recipe.update(raw_threshold=chosen, calibrated_threshold=float(calibrate(chosen, recipe)))
    return recipe


def event_two(y, score, ids):
    if len(y) != 4000 or len(np.unique(ids)) != len(y):
        raise ValueError("The challenge statistic expects exactly 4000 unique stay keys")
    # Follow score.c v1.4, including the risk/RecordID order and denominator.
    order = np.lexsort((ids, score)).reshape(10, 400)
    rate = score[order].mean(axis=1)
    observed = y[order].sum(axis=1)
    statistic = np.sum((observed - 400 * rate) ** 2 / (400 * (rate * (1 - rate) + 0.001)))
    spread = rate[-1] - rate[0]
    return float(statistic / spread) if spread > 0 else None


def assess(y, scores, recipes, codes, repeats, seed, ids=None):
    results = {}
    delta = {}
    for tag, recipe in recipes.items():
        raw = scores[tag]
        calibrated = calibrate(raw, recipe)
        before, after = score_metrics(y, raw), score_metrics(y, calibrated)
        np.testing.assert_allclose([before["ap"], before["auroc"]], [after["ap"], after["auroc"]], rtol=0, atol=1e-12)
        # XGBoost scores are float32. Compare in float64 to keep a frozen threshold exact.
        raw = raw.astype(np.float64)
        np.testing.assert_array_equal(raw >= recipe["raw_threshold"], calibrated >= recipe["calibrated_threshold"])
        results[tag] = {
            "raw": before, "calibrated": after,
            "raw_0_5": binary_metrics(y, raw, 0.5), "calibrated_0_5": binary_metrics(y, calibrated, 0.5),
            "ab_threshold": binary_metrics(y, raw, recipe["raw_threshold"]),
        }
        if ids is not None:
            results[tag]["event_2_raw"] = event_two(y, raw, ids)
            results[tag]["event_2_calibrated"] = event_two(y, calibrated, ids)
        delta[tag] = np.bincount(codes, weights=np.square(calibrated - y) - np.square(raw - y))
    count = np.bincount(codes)
    rng = np.random.default_rng(seed)
    bootstrap = {name: [] for name in recipes}
    for _ in range(repeats):
        sample = rng.integers(len(count), size=len(count))
        n = count[sample].sum()
        for tag, values in delta.items():
            bootstrap[tag].append(values[sample].sum() / n)
    for tag, values in bootstrap.items():
        results[tag]["paired_brier_ci"] = np.quantile(values, (0.025, 0.975)).tolist()
    return results


def development(stage):
    x, y, folds = ab_inputs()
    p33_path, p38_path = P33 / "fitted/manifest.json", ABLATION / "fitted/manifest.json"
    p33, p38 = map(from_json, (p33_path, p38_path))
    paths = [AB_FEATURES, AB_FOLDS, AB_LABELS, AB_SELECTION, p33_path, p38_path]
    protected = {**p38["input_sha256"], **{path.relative_to(OUT).as_posix(): sha(path) for path in paths}}
    probability = {}
    selection = from_json(AB_SELECTION)
    for tag, key in ORIGINALS.items():
        root = OUT / "runs" / selection[key]
        score_path = root / "oof.csv"
        archived = pd.read_csv(score_path, index_col="RecordID")
        if not archived.index.is_unique or set(archived.index) != set(x.index):
            raise ValueError(f"Expected original A/B keys for {tag}")
        restored = np.full(len(y), np.nan)
        for fold in FOLD_IDS:
            model_path = root / f"fold{fold}.joblib"
            protected[model_path.relative_to(OUT).as_posix()] = sha(model_path)
            pipe, columns = verified_template(model_path, protected[AB_FEATURES.relative_to(OUT).as_posix()])
            mask = folds == fold
            validate_oof(pipe, columns, x.loc[mask], y[mask], fold, archived)
            restored[mask] = no_category_warning(pipe, x.loc[mask, columns])
        protected[score_path.relative_to(OUT).as_posix()] = sha(score_path)
        probability[tag] = restored
    verify_sources(protected)
    packet = stage / "ab_replay.npz"
    worker = Path(__file__).with_name("replay_calibration_ab.py")
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, (P33 / "cpu_deps", worker.parent))),
    }
    subprocess.run([str(PYTORCH), str(worker), str(packet)], check=True, env=env)
    with np.load(packet, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["y"], y)
        np.testing.assert_array_equal(result["folds"], folds)
        probability.update({tag: result[tag] for tag in ("p28", "p33", "p38")})
    reference = {
        "p28": p38["scores"]["p28"]["clean"],
        "p33": p33["metrics"][p33["winner"]]["clean"],
        "p38": p38["scores"][p38["winner"]]["clean"],
    }
    for tag, point in reference.items():
        actual = np.mean([average_precision_score(y[folds == f], probability[tag][folds == f]) for f in FOLD_IDS])
        np.testing.assert_allclose(actual, point, rtol=0, atol=1e-12)
    verify_sources(protected)
    recipes = {tag: learn(y, score) for tag, score in probability.items()}
    stats = {tag: binary_metrics(y, probability[tag], item["raw_threshold"]) for tag, item in recipes.items()}
    return recipes, stats, protected, x.columns.tolist()


def demo_scores(stage, columns, protected):
    routed, _, gcs, extras, sources = frozen_models()
    rule, elastic, reduced = frozen_ensemble(sources)
    protected.update(sources)
    paths = {
        "p28": P28_EVAL / "summary.json", "p33": P33 / "evaluation/summary.json",
        "p38": ABLATION / "evaluation/summary.json", "p23": OUT / "p23_eicu_explicit_urine/summary.json",
    }
    history = {key: from_json(path) for key, path in paths.items()}
    for path in paths.values():
        protected[path.relative_to(OUT).as_posix()] = sha(path)
    model33, model38 = P33 / "fitted/sequence.pt", ABLATION / "fitted/sequence.pt"
    for tag, path in (("p33", model33), ("p38", model38)):
        protected[path.relative_to(OUT).as_posix()] = history[tag]["training"]["model_sha256"]
    verify_sources(protected)
    cohort, cases, batch, hashes = eicu_sequence_cases(history["p28"], columns, extras)
    y = cohort.y.to_numpy()
    original_full, full_cols = benchmarks()["frozen_full"]
    p38 = history["p38"]["training"]
    keep = [GROUPS.index(name) for name in p38["included_channels"]]
    forecasts33 = infer(stage, model33, **batch)
    forecasts38 = infer(stage, model38, four_hour=True, channels=keep, **batch)
    anchor = Ensemble(routed, gcs, rule)
    archived = {tag: {(r["alignment"], r["mode"]): r for r in history[tag]["eicu"]} for tag in ("p28", "p33", "p38")}
    old_full = {r["mode"]: r for r in history["p23"]["rows"] if r["model"] == "Full" and r["variant"] == "frozen"}
    results = {}
    for i, (alignment, mode, frame, _) in enumerate(cases):
        table = anchor.predict_proba(frame)[:, 1]
        scores = {
            "frozen_full": no_category_warning(original_full, frame.loc[:, full_cols]),
            "frozen_elastic": no_category_warning(elastic, frame.loc[:, reduced]),
            "p28": table, "p33": 0.75 * table + 0.25 * forecasts33[i].astype(float),
            "p38": blend(table, forecasts38[i], p38["selected"]),
        }
        key = alignment, mode
        for tag in ("p28", "p33", "p38"):
            checked_metrics(y, scores[tag], archived[tag][key]["metrics"])
        checked_metrics(y, scores["frozen_elastic"], archived["p38"][key]["frozen_elastic"])
        checked_metrics(y, scores["frozen_full"], old_full[mode][alignment])
        results[f"{alignment}:{mode}"] = scores
    if source_hashes(DEMO) != hashes:
        raise ValueError("The original Demo inputs changed")
    return results, cohort, hashes


def evaluate():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p42-calibration-") as temp:
        stage = Path(temp)
        recipes, development_metrics, protected, columns = development(stage)
        print("Froze all A/B calibration parameters and decision thresholds", flush=True)
        raw, y, predictions, _, sources, mask_digest = replay_frozen_c(stage)
        protected.update(sources)
        c = {
            scenario: assess(y, scores, recipes, np.arange(len(y)), C_BOOTSTRAPS, C_BOOTSTRAP_SEED, raw.index.to_numpy())
            for scenario, scores in predictions.items()
        }
        print("Scored calibrated and raw risks on all three frozen C views", flush=True)
        predictions, cohort, hashes = demo_scores(stage, columns, protected)
        codes = pd.factorize(cohort.uniquepid, sort=False)[0]
        if (codes < 0).any():
            raise ValueError("Expected patient keys for paired eICU resampling")
        eicu = {key: assess(cohort.y.to_numpy(), values, recipes, codes, BOOTSTRAPS, SEED)
                for key, values in predictions.items()}
        verify_sources(protected)
        output = stage / ROOT.name
        output.mkdir()
        save_json(output / "summary.json", {
            "device": "cpu", "a_b_stays": 8000, "a_b_deaths": 1122,
            "original_network_fold_replays": 6, "new_prediction_model_fits": 0, "calibration_fits": len(recipes),
            "recipes": recipes, "a_b_threshold_metrics": development_metrics, "set_c": c, "eicu": eicu,
            "source_sha256": protected, "raw_eicu_sha256": hashes, "set_c_mask_sha256": mask_digest,
            "set_c_n_bootstrap": C_BOOTSTRAPS, "eicu_n_bootstrap": BOOTSTRAPS,
        })
        output.rename(ROOT)
    print(f"P42 CPU calibration and threshold audit saved to {ROOT}", flush=True)


if __name__ == "__main__":
    evaluate()
