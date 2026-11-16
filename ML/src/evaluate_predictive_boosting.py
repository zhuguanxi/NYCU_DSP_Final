"""Evaluate A/B-selected predictors beside frozen XGBoost benchmarks."""

import json
import tempfile
import warnings
from pathlib import Path

from train_predictive_boosting import (
    AB_FEATURES, ANCHORS, FROZEN, GCS_ONLY_OUTPUT, OUTPUT, native_inputs, native_predict, verify_sources,
)

import joblib
import numpy as np
import pandas as pd

from evaluate_eicu_demo import MODES, ranking
from evaluate_ma_gbt_p12 import KEYS, P11, REFERENCES, eicu_inputs, plan_for
from evaluate_missingness_augmentation import augmented_models, paired_intervals
from prepare import OUT, sha
from stress_missingness_c import C_FEATURES, C_LABELS, C_PREDICTIONS, COVERAGE, METRICS, load_cohort, score_metrics, stress
from train_missingness_augmentation import from_json

EVALUATION = OUTPUT / "evaluation"
C_BOOTSTRAPS = 2000
C_BOOTSTRAP_SEED = 20261013
COMPARATORS = (*ANCHORS, "gcs_only_elastic")
WINS = ("clean_selected", "balanced_selected")


class Predictor:
    def __init__(self, components, models, definitions):
        self.components = components
        self.models = models
        self.definitions = definitions

    def predict_proba(self, frame):
        result = np.zeros(len(frame), dtype=float)
        for key, weight in self.components.items():
            model = self.models[key]
            if key in ANCHORS:
                pipe, columns = model
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Found unknown categories.*")
                    score = pipe.predict_proba(frame.loc[:, columns])[:, 1]
            else:
                spec = self.definitions[key]
                score = native_predict(model, spec, native_inputs(frame, spec["scope"]), spec["rounds"])
            result += weight * score
        return np.column_stack((1 - result, result))


def no_category_warning(pipe, frame):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        return pipe.predict_proba(frame)[:, 1]


def benchmarks():
    original, _, cols = joblib.load(FROZEN / "Full.joblib")
    original.named_steps["model"].set_params(device="cpu", n_jobs=2)
    _, gcs, manifest = augmented_models(GCS_ONLY_OUTPUT)
    if manifest["selected_rates_percent"] != {"GCS": 60}:
        raise ValueError("Unexpected GCS-only baseline")
    return {
        "frozen_full": (original, cols),
        "gcs_only_full": gcs["Full"],
        "gcs_only_elastic": gcs["Elastic"],
    }


def contrast_c(y, baseline, selected):
    rng = np.random.default_rng(C_BOOTSTRAP_SEED)
    comparisons = {pick: {control: [] for control in baseline} for pick in selected}
    for _ in range(C_BOOTSTRAPS):
        indices = rng.integers(len(y), size=len(y))
        target = y[indices]
        if target.min() == target.max():
            raise ValueError("Set C bootstrap contains one class")
        old = {key: ranking(target, pred[indices]) for key, pred in baseline.items()}
        for pick, pred in selected.items():
            point = np.asarray(ranking(target, pred[indices]))
            for tag, reference in old.items():
                comparisons[pick][tag].append(point - reference)
    return {
        pick: {tag: dict(zip(("ap", "auroc"), np.quantile(values, [0.025, 0.975], axis=0).T.tolist(), strict=True))
               for tag, values in comparisons[pick].items()}
        for pick in comparisons
    }


def compare_c(pipes, controls, reference, prior):
    frame, y, historical = load_cohort()
    positions, plan, draws_hash = plan_for(frame)
    if draws_hash != reference["mask_draws_sha256"]:
        raise ValueError("Set C mask draws differ from P9")
    new_scores = {key: model.predict_proba(frame.loc[:, cols])[:, 1] for key, (model, cols) in pipes.items()}
    ref_scores = {key: no_category_warning(pipe, frame.loc[:, cols]) for key, (pipe, cols) in controls.items()}
    np.testing.assert_array_equal(
        ref_scores["frozen_full"], historical["score_Full"].to_numpy(dtype=ref_scores["frozen_full"].dtype),
    )
    baseline, result = stress(
        frame, y, pd.DataFrame({f"score_{key}": pred for key, pred in new_scores.items()}, index=frame.index),
        pipes, positions, plan,
    )
    tag_columns = {"frozen_full": ("Full", "frozen"), "gcs_only_full": ("Full", "gcs"), "gcs_only_elastic": ("Elastic", "gcs")}
    history = {(row["scenario"], row["model"]): row for row in prior["set_c"]}
    old_c = pd.read_csv(P11 / "set_c_three_way_replicates.csv")
    mask_keys = [key for key in KEYS if key != "model"]
    full = {}
    for key, (feature_set, tag) in tag_columns.items():
        values = score_metrics(y, ref_scores[key])
        np.testing.assert_allclose(
            [values[m] for m in METRICS],
            [history["clean", feature_set][f"{m}_{tag}"] for m in METRICS], rtol=0, atol=1e-12,
        )
        full[key] = old_c.loc[old_c.model.eq(feature_set), mask_keys + [f"{m}_{tag}" for m in METRICS]].rename(
            columns={f"{m}_{tag}": f"{m}_{key}" for m in METRICS}
        )
        result = result.merge(full[key], on=mask_keys, validate="many_to_one")
    if len(result) != 168 * len(pipes) or any(len(ref) != 168 for ref in full.values()):
        raise ValueError("Set C case counts differ from P9/P11")
    rows = []
    for pick, actual in baseline.items():
        row = {"scenario": "clean", "model": pick, "repeats": 1, **actual}
        for tag in COMPARATORS:
            for metric, value in score_metrics(y, ref_scores[tag]).items():
                row[f"{metric}_{tag}"] = value
                row[f"delta_{metric}_vs_{tag}"] = actual[metric] - value
        rows.append(row)
    for (scenario, model), group in result.groupby(["scenario", "model"], sort=False):
        aggregate = {"scenario": scenario, "model": model, "repeats": len(group)}
        for metric in METRICS:
            aggregate[metric] = float(group[metric].mean())
            for tag in COMPARATORS:
                col = f"{metric}_{tag}"
                delta = group[metric] - group[col]
                aggregate[col] = float(group[col].mean())
                aggregate[f"delta_{metric}_vs_{tag}"] = float(delta.mean())
                aggregate[f"delta_{metric}_vs_{tag}_p10_p90"] = delta.quantile([0.1, 0.9]).tolist()
        rows.append(aggregate)
    intervals = contrast_c(y, ref_scores, new_scores)
    return rows, result, intervals


def compare_eicu(pipes, reference, prior, inputs=None):
    cohort, frames = eicu_inputs() if inputs is None else inputs
    y = cohort.y.to_numpy()
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    if sha(gcs_path) != reference["gcs_scores_sha256"]:
        raise ValueError("Frozen GCS-only eICU predictions changed")
    frozen_path = OUT / "eicu_demo_external" / "predictions.parquet"
    original, gcs = pd.read_parquet(frozen_path), pd.read_parquet(gcs_path)
    if not all(scores.index.equals(cohort.index) for scores in (original, gcs)):
        raise ValueError("Saved eICU probabilities do not align")
    controls = {
        "frozen_full": {mode: original[f"{mode}_Full"].to_numpy() for mode in MODES},
        "gcs_only_full": {mode: gcs[f"{mode}_Full"].to_numpy() for mode in MODES},
        "gcs_only_elastic": {mode: gcs[f"{mode}_Elastic"].to_numpy() for mode in MODES},
    }
    history = {(row["mode"], row["model"]): row for row in prior["eicu"]}
    for tag, (name, column) in {
        "frozen_full": ("Full", "frozen"), "gcs_only_full": ("Full", "gcs"), "gcs_only_elastic": ("Elastic", "gcs"),
    }.items():
        for mode, scores in controls[tag].items():
            stats = score_metrics(y, scores)
            np.testing.assert_allclose(
                [stats[m] for m in METRICS], [history[mode, name][f"{m}_{column}"] for m in METRICS],
                rtol=0, atol=1e-12,
            )
    print("eICU original cadences and patient-level paired bootstrap", flush=True)
    rows = []
    for pick, (predictor, cols) in pipes.items():
        new = {mode: predictor.predict_proba(frame.loc[:, cols])[:, 1] for mode, frame in frames.items()}
        intervals = {tag: paired_intervals(cohort, values, new) for tag, values in controls.items()}
        for mode, scores in new.items():
            measures = score_metrics(y, scores)
            row = {"mode": mode, "model": pick, **measures}
            for tag, values in controls.items():
                for metric, reference_value in score_metrics(y, values[mode]).items():
                    row[f"{metric}_{tag}"] = reference_value
                    row[f"delta_{metric}_vs_{tag}"] = measures[metric] - reference_value
                    if metric in intervals[tag][mode]:
                        row[f"delta_{metric}_vs_{tag}_ci"] = intervals[tag][mode][metric]
            rows.append(row)
    return rows


def evaluate():
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    manifest_path = OUTPUT / "manifest.json"
    fit = from_json(manifest_path)
    if (fit["device"], fit["dataset"]) != ("cpu", "P12 A/B"):
        raise ValueError("Evaluation requires frozen A/B CPU selection")
    reference_path = REFERENCES / "manifest.json"
    reference = from_json(reference_path)
    prior = from_json(P11 / "summary.json")
    extra = (manifest_path, reference_path, C_FEATURES, C_LABELS, C_PREDICTIONS, COVERAGE)
    protected = {
        **fit["source_sha256"], **reference["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in extra},
        **{(OUTPUT / f"{tag}.joblib").relative_to(OUT).as_posix(): digest for tag, digest in fit["model_sha256"].items()},
        (REFERENCES / "gcs_eicu_scores.parquet").relative_to(OUT).as_posix(): reference["gcs_scores_sha256"],
    }
    verify_sources(protected)
    controls = benchmarks()
    models = {**controls, **{tag: joblib.load(OUTPUT / f"{tag}.joblib") for tag in fit["model_sha256"]}}
    cols = pd.read_parquet(AB_FEATURES).columns.tolist()
    pipes = {key: (Predictor(fit["selected_components"][key], models, fit["native_definitions"]), cols) for key in WINS}
    set_c, paired, set_c_ci = compare_c(pipes, controls, reference, prior)
    eicu = compare_eicu(pipes, reference, prior)
    verify_sources(protected)
    report = {
        "training": fit, "set_c": set_c, "set_c_clean_ci": set_c_ci, "eicu": eicu,
        "c_bootstraps": C_BOOTSTRAPS, "source_sha256": protected,
    }
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="p21-eval-") as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        paired.to_csv(staging / "set_c_paired_replicates.csv", index=False)
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(EVALUATION)
    print(f"P21 frozen evaluation saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    evaluate()
