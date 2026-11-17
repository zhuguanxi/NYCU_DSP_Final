"""Isolate the two P44 routes with the original folded external comparison."""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from audit_p12_outcome_timing import ap_intervals
from eicu_demo_features import DEMO
from evaluate_eicu_confirmation import frozen_ensemble, frozen_models
from evaluate_eicu_demo import BOOTSTRAPS, SEED, source_hashes
from evaluate_gcs_routed_temporal import RoutedPredictor, checked_metrics
from evaluate_ma_gbt_p12 import REFERENCES, plan_for
from evaluate_real_eicu_augmentation import BRANCHES, HOSPITAL_SHA256, OUTPUT as P44, demo_setup, train_pair
from evaluate_robust_ensemble import EVALUATION as P28, Ensemble
from evaluate_temporal_features import save_json, set_c_temporal
from pilot_temporal_features import DEST as P22
from prepare import OUT, sha
from stress_missingness_c import METRICS, SCENARIOS, apply_masks, load_cohort, mask_draws, score_metrics
from train_predictive_boosting import ab_inputs, from_json, verify_sources

OUTPUT = OUT / "p45_augmentation_branch_ablation"
ROUTES = {
    "primary_only": ("temporal",),
    "fallback_only": ("blind_lgb",),
}
PAIRS = (("primary_only", "p28"), ("fallback_only", "p28"), ("primary_only", "p44"), ("p44", "p28"))


def mix_routes(original, augmented, frame):
    absent = frame["GCS__count"].eq(0).to_numpy()
    return {
        "p28": original, "p44": augmented,
        "primary_only": np.where(absent, original, augmented),
        "fallback_only": np.where(absent, augmented, original),
    }


def recorded_c(root):
    saved = from_json(root / "summary.json")
    baseline = next(row for row in saved["set_c"] if row["scenario"] == "clean")
    replicates = pd.read_csv(root / "set_c_paired_replicates.csv")
    lookup = {(row["scenario"], row["repeat"]): row for row in replicates.to_dict("records")}
    if len(lookup) != 168 or len(replicates) != 168:
        raise ValueError("Expected the original 168 Set C masks")
    return baseline, lookup


def evaluate_c(routed, gcs, rule, fitted, protected):
    raw, y, _ = load_cohort()
    paths = [root / "set_c_paired_replicates.csv" for root in (P28, P44)]
    reference_path = REFERENCES / "manifest.json"
    protected.update({path.relative_to(OUT).as_posix(): sha(path) for path in (*paths, reference_path)})
    controls = {tag: recorded_c(root) for tag, root in (("p28", P28), ("p44", P44))}
    positions, plan, digest = plan_for(raw)
    if digest != from_json(reference_path)["mask_draws_sha256"]:
        raise ValueError("Expected the original Set C missingness plan")
    timing = set_c_temporal(raw.index)
    base = Ensemble(RoutedPredictor(routed.rule, routed.models, raw, timing), gcs, rule)
    both = Ensemble(RoutedPredictor(routed.rule, {**routed.models, **fitted}, raw, timing), gcs, rule)

    def score(frame, scenario, repeat):
        values = mix_routes(base.predict_proba(frame)[:, 1], both.predict_proba(frame)[:, 1], frame)
        for tag, (prior, draws) in controls.items():
            checked_metrics(y, values[tag], prior if scenario == "clean" else draws[scenario, repeat])
        return values

    clean = score(raw, "clean", 1)
    for key, branches in ROUTES.items():
        altered = {**routed.models, **{name: fitted[name] for name in branches}}
        reference = Ensemble(RoutedPredictor(routed.rule, altered, raw, timing), gcs, rule)
        np.testing.assert_array_equal(reference.predict_proba(raw)[:, 1], clean[key])
    scenarios = {"clean": {tag: score_metrics(y, value) for tag, value in clean.items()}}
    intervals = {"clean": ap_intervals(y, clean, PAIRS)}
    draws = mask_draws(positions, plan)
    with threadpool_limits(limits=2):
        for scenario, variables, design in SCENARIOS:
            replicates = []
            for repeat, draw in enumerate(draws if design == "matched" else [{}], start=1):
                masked = apply_masks(raw, {name: draw.get(name, positions[name]) for name in variables}, plan, design)
                values = score(masked, scenario, repeat)
                replicates.append({tag: score_metrics(y, value) for tag, value in values.items()})
            scenarios[scenario] = {
                tag: {metric: float(np.mean([r[tag][metric] for r in replicates])) for metric in METRICS} for tag in clean
            }
            if scenario == "full_GCS":
                np.testing.assert_array_equal(values["primary_only"], values["p28"])
                np.testing.assert_array_equal(values["fallback_only"], values["p44"])
            if scenario in {"full_GCS", "full_all_six"}:
                intervals[scenario] = ap_intervals(y, values, PAIRS)
            print(f"P45 Set C {scenario}: reproduced both controls, {len(replicates)} mask draws", flush=True)
    verify_sources(protected)
    return scenarios, intervals, digest


def external_intervals(cohort, scores):
    y = cohort.y.to_numpy()
    groups = list(cohort.groupby("uniquepid", sort=False).indices.values())
    samples = {f"{new}_vs_{old}": [] for new, old in PAIRS}
    rng = np.random.default_rng(SEED)
    for _ in range(BOOTSTRAPS):
        drawn = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in drawn])
        outcome = y[idx]
        if outcome.min() == outcome.max():
            continue
        ranks = {tag: average_precision_score(outcome, value[idx]) for tag, value in scores.items()}
        for new, old in PAIRS:
            samples[f"{new}_vs_{old}"].append(float(ranks[new] - ranks[old]))
    if any(len(rows) < BOOTSTRAPS * 0.95 for rows in samples.values()):
        raise ValueError("Too few valid patient draws")
    return {tag: {"ap": np.quantile(rows, (0.025, 0.975)).tolist(), "n": len(rows)} for tag, rows in samples.items()}


def evaluate_demo(routed, gcs, rule, extras, prior, old38, p44):
    x, y, _ = ab_inputs()
    timing = pd.read_parquet(P22 / "features_AB_temporal.parquet")
    if not timing.index.equals(x.index) or timing.columns.tolist() != extras:
        raise ValueError("Unexpected A/B feature order")
    cohort, frames, extra, hashes, folds, sizes, grouping, provenance = demo_setup(x, extras, prior, old38)
    if (
        sizes != p44["folds"] or grouping != p44["grouping"] or provenance != p44["demo_identifier_sources"]
        or hashes != p44["raw_eicu_sha256"]
    ):
        raise ValueError("The Demo patient partition or source differs from P44")
    y_demo = cohort.y.to_numpy()
    original = Ensemble(routed, gcs, rule)
    p28 = {key: original.predict_proba(frame)[:, 1] for key, frame in frames.items()}
    augmented = {key: np.full(len(cohort), np.nan) for key in frames}
    for index, (train, held) in enumerate(folds, start=1):
        current = train_pair(x, timing, y, extra, train, y_demo)
        both = Ensemble(RoutedPredictor(routed.rule, {**routed.models, **current}), gcs, rule)
        for key, frame in frames.items():
            augmented[key][held] = both.predict_proba(frame.iloc[held])[:, 1]
        print(f"P45 reproduced both P44 Demo branches on fold {index}/{len(folds)}", flush=True)
    results = {}
    for key, frame in frames.items():
        case = ":".join(key)
        values = mix_routes(p28[key], augmented[key], frame)
        checked_metrics(y_demo, values["p28"], p44["eicu"][case]["p28"])
        checked_metrics(y_demo, values["p44"], p44["eicu"][case]["metrics"])
        intervals = external_intervals(cohort, values)
        expected = p44["eicu"][case]["paired_ci_vs_p28"]
        np.testing.assert_allclose(intervals["p44_vs_p28"]["ap"], expected["ap"], atol=1e-12, rtol=0)
        absent = frame["GCS__count"].eq(0).to_numpy()
        strata = {}
        for label, mask in (("observed_gcs", ~absent), ("absent_gcs", absent)):
            truth = y_demo[mask]
            strata[label] = {
                "stays": len(truth), "deaths": int(truth.sum()),
                "metrics": {tag: score_metrics(truth, score[mask]) for tag, score in values.items()},
            }
        results[case] = {
            "metrics": {tag: score_metrics(y_demo, score) for tag, score in values.items()},
            "paired_ap_ci": intervals, "strata": strata,
        }
    return results, sizes, hashes


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    routed, _, gcs, extras, protected = frozen_models()
    rule, _, _ = frozen_ensemble(protected)
    prior_path, p44_path = P28 / "summary.json", P44 / "summary.json"
    old38_path = OUT / "p38_sequence_channel_ablation/evaluation/summary.json"
    prior, p44, old38 = map(from_json, (prior_path, p44_path, old38_path))
    if routed.rule != {"primary": "temporal", "fallback": "blind_lgb", "weight": 1.0} or (
        p44["p26_rule"], p44["p28_rule"], p44["device"], p44["extra_weight_per_stay"]
    ) != (routed.rule, rule, "cpu", 1.0):
        raise ValueError("Expected the original P44 and P28 routing recipes")
    model_paths = {tag: P44 / "fitted" / f"{tag}.joblib" for tag in BRANCHES}
    protected.update(p44["source_sha256"])
    protected.update({path.relative_to(OUT).as_posix(): sha(path) for path in (prior_path, p44_path, old38_path)})
    protected.update({path.relative_to(OUT).as_posix(): p44["model_sha256"][tag] for tag, path in model_paths.items()})
    verify_sources(protected)
    fitted = {tag: joblib.load(path) for tag, path in model_paths.items()}
    eicu, sizes, hashes = evaluate_demo(routed, gcs, rule, extras, prior, old38, p44)
    c, c_ci, mask_digest = evaluate_c(routed, gcs, rule, fitted, protected)
    verify_sources(protected)
    if hashes != source_hashes(DEMO) or sha(DEMO / "hospital.csv.gz") != HOSPITAL_SHA256:
        raise ValueError("Demo source changed during branch evaluation")
    with tempfile.TemporaryDirectory(dir=OUT, prefix="branch-ablation-") as tmp:
        stage = Path(tmp) / OUTPUT.name
        stage.mkdir()
        save_json(stage / "summary.json", {
            "device": "cpu", "final_fits": 0, "original_demo_branch_replays": len(sizes) * len(BRANCHES),
            "branches_from_p44": ROUTES, "demo_folds": sizes, "eicu": eicu, "set_c": c, "set_c_paired_ap_ci": c_ci,
            "set_c_mask_sha256": mask_digest, "source_sha256": protected, "raw_eicu_sha256": hashes,
        })
        stage.rename(OUTPUT)
    print(f"P45 branch ablation evaluated at {OUTPUT}", flush=True)


if __name__ == "__main__":
    evaluate()
