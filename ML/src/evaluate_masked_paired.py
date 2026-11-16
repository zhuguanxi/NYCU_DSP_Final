"""Estimate paired Set C AP and AUROC differences under fixed P9 full masks."""

import tempfile
from pathlib import Path

from evaluate_eicu_confirmation import frozen_ensemble, frozen_models

import pandas as pd

from evaluate_gcs_routed_temporal import EVALUATION as P26, RoutedPredictor, checked_metrics
from evaluate_ma_gbt_p12 import P11, REFERENCES, plan_for
from evaluate_predictive_boosting import C_BOOTSTRAPS, C_BOOTSTRAP_SEED, benchmarks, contrast_c, no_category_warning
from evaluate_robust_ensemble import EVALUATION as P28, Ensemble
from evaluate_temporal_features import save_json, set_c_temporal
from prepare import OUT, sha
from stress_missingness_c import METRICS, SCENARIOS, apply_masks, load_cohort, score_metrics
from train_predictive_boosting import AB_FEATURES, from_json, verify_sources

OUTPUT = OUT / "p30_masked_paired"
SCENARIO_NAMES = ("full_GCS", "full_all_six")
CONTROLS = ("frozen_full", "gcs_only_full", "gcs_only_elastic")


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    routed, _, elastic, extras, protected = frozen_models()
    rule, _, _ = frozen_ensemble(protected)
    p26, p28, p11, reference_path = (P26 / "summary.json", P28 / "summary.json", P11 / "summary.json", REFERENCES / "manifest.json")
    prior_routed, prior_ensemble, prior_controls, references = map(from_json, (p26, p28, p11, reference_path))
    protected.update(prior_routed["source_sha256"])
    protected.update(prior_ensemble["source_sha256"])
    protected.update({path.relative_to(OUT).as_posix(): sha(path) for path in (p26, p28, p11, reference_path)})
    verify_sources(protected)

    raw, y, _ = load_cohort()
    positions, plan, mask_digest = plan_for(raw)
    if mask_digest != references["mask_draws_sha256"]:
        raise ValueError("P9 mask plan differs from frozen comparator evaluation")
    timing = set_c_temporal(raw.index)
    cols = pd.read_parquet(AB_FEATURES).columns.tolist()
    if list(timing.columns) != extras or not timing.index.equals(raw.index):
        raise ValueError("Set C temporal feature schema changed")
    routed_c = RoutedPredictor(routed.rule, routed.models, raw, timing)
    ensemble = Ensemble(routed_c, elastic, rule)
    controls = benchmarks()
    pipes = {**controls, "routed": (routed_c, cols), "ensemble": (ensemble, cols)}
    history = {(row["scenario"], row["model"]): row for row in prior_controls["set_c"]}
    source_entry = {
        "frozen_full": ("Full", "frozen"), "gcs_only_full": ("Full", "gcs"),
        "gcs_only_elastic": ("Elastic", "gcs"),
    }
    scenarios = {"clean": raw}
    for name, variables, design in SCENARIOS:
        if name in SCENARIO_NAMES:
            scenarios[name] = apply_masks(raw, {key: positions[key] for key in variables}, plan, design)
    results = {}
    for name, frame in scenarios.items():
        predicted = {key: no_category_warning(pipe, frame.loc[:, columns]) for key, (pipe, columns) in pipes.items()}
        for key, (feature_set, prefix) in source_entry.items():
            saved = history[name, feature_set]
            checked_metrics(y, predicted[key], {metric: saved[f"{metric}_{prefix}"] for metric in METRICS})
        for key, previous in (("routed", prior_routed), ("ensemble", prior_ensemble)):
            expected, = [row for row in previous["set_c"] if row["scenario"] == name]
            checked_metrics(y, predicted[key], expected)
        metrics = {key: score_metrics(y, value) for key, value in predicted.items()}
        if name == "clean":
            clean = predicted
            continue
        controls_scores = {key: predicted[key] for key in (*CONTROLS, "routed")}
        selected_scores = {key: predicted[key] for key in ("routed", "ensemble")}
        for key in selected_scores:
            controls_scores[f"{key}_clean"] = clean[key]
        intervals = contrast_c(y, controls_scores, selected_scores)
        contrasts = {}
        for key, comparisons in intervals.items():
            refs = (*CONTROLS, f"{key}_clean")
            if key == "ensemble":
                refs += ("routed",)
            contrasts[key] = {
                comparator: {
                    "delta": {
                        metric: metrics[key][metric] - score_metrics(y, controls_scores[comparator])[metric]
                        for metric in ("ap", "auroc")
                    },
                    "ci": comparisons[comparator],
                }
                for comparator in refs
            }
        results[name] = {"metrics": metrics, "contrasts": contrasts}
        print(f"Replayed masked point estimates and computed paired intervals for {name}", flush=True)

    verify_sources(protected)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p30-paired-") as temporary:
        stage = Path(temporary) / OUTPUT.name
        stage.mkdir()
        save_json(stage / "summary.json", {
            "device": "cpu", "fits": 0, "stays": len(y), "deaths": int(y.sum()),
            "bootstrap_repeats": C_BOOTSTRAPS, "seed": C_BOOTSTRAP_SEED,
            "scenarios": results, "source_sha256": protected,
        })
        stage.rename(OUTPUT)


if __name__ == "__main__":
    run()
