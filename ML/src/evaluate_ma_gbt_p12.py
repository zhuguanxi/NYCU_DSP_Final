"""Evaluate frozen A/B MA-GBT on the original Set C and eICU Demo protocol."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from ma_gbt_p12_inputs import RUN, SCHEMA_FILE, digests, encode

import joblib
import numpy as np
import pandas as pd

from evaluate_eicu_demo import BOOTSTRAPS, MODES, MODELS, P7_PROTOCOL, SEED, THRESHOLD
from evaluate_eicu_demo import evaluate_predictions, score_frozen
from evaluate_missingness_augmentation import EICU, P9, augmented_models, paired_intervals
from prepare import OUT, sha
from stress_missingness_c import C_FEATURES, C_LABELS, C_PREDICTIONS, COVERAGE, METRICS, REPEATS
from stress_missingness_c import SEED as MASK_SEED
from stress_missingness_c import load_cohort, mask_draws, mask_plan, score_metrics, stress
from train_missingness_augmentation import GCS_ONLY_OUTPUT, from_json

FITTED = RUN / "fitted"
REFERENCES = RUN / "reference"
OUTPUT = RUN / "evaluation"
P11 = OUT / "gcs_only_comparison"
KEYS = ["scenario", "model", "repeat", "design", "variables"]
TAGS = ("frozen", "gcs")


class Predictor:
    def __init__(self, model, schema):
        self.model = model
        self.schema = schema

    def predict_proba(self, frame):
        features, _ = encode(frame, self.schema)
        return self.model.predict_proba(features)

    def predict_and_rho(self, frame):
        features, mask = encode(frame, self.schema)
        return self.model.predict_proba(features)[:, 1], float(
            self.model.compute_missingness_reliance(features, mask)
        )


def eicu_inputs():
    cohort = pd.read_parquet(EICU / "cohort.parquet")
    frames = {mode: pd.read_parquet(EICU / f"features_{mode}.parquet") for mode in MODES}
    if not cohort.index.is_unique or not all(frame.index.equals(cohort.index) for frame in frames.values()):
        raise ValueError("eICU feature and outcome rows differ")
    saved = from_json(EICU / "summary.json")
    if (len(cohort), cohort.uniquepid.nunique(), int(cohort.y.sum())) != (saved["stays"], saved["patients"], saved["deaths"]):
        raise ValueError("eICU cohort changed")
    if (saved["bootstrap_seed"], saved["bootstrap_replicates"]) != (SEED, BOOTSTRAPS):
        raise ValueError("eICU bootstraps differ from P8")
    return cohort, frames


def plan_for(features):
    original = from_json(P9 / "summary.json")
    if (original["seed"], original["matched_repeats"], original["threshold"]) != (MASK_SEED, REPEATS, THRESHOLD):
        raise ValueError("P9 masking settings differ")
    for path in (C_FEATURES, C_LABELS, C_PREDICTIONS, COVERAGE):
        if sha(path) != original["source_sha256"][path.name]:
            raise ValueError(f"P9 input changed: {path.name}")
    positions, plan = mask_plan(features)
    if plan != original["mask_plan"]:
        raise ValueError("P9 mask target counts differ")
    digest = hashlib.sha256()
    for replicate in mask_draws(positions, plan):
        for variable, indices in replicate.items():
            digest.update(variable.encode())
            digest.update(indices.astype("<i8").tobytes())
    return positions, plan, digest.hexdigest()


def check_fit():
    saved = from_json(FITTED / "manifest.json")
    if saved["training_dataset"] != "P12 A/B" or saved["device"] != "cpu":
        raise ValueError("Expected frozen CPU MA-GBT fitted on A/B")
    if sha(SCHEMA_FILE) != saved["encoded_schemas_sha256"]:
        raise ValueError("Frozen training schema changed")
    for name in MODELS:
        if sha(FITTED / f"{name}.joblib") != saved["model_sha256"][name]:
            raise ValueError(f"Frozen MA-GBT model changed: {name}")
    return saved


def export_references():
    check_fit()
    if REFERENCES.exists():
        raise FileExistsError(REFERENCES)
    p11 = from_json(P11 / "summary.json")
    frozen_hashes = from_json(P7_PROTOCOL)["model_sha256"]
    gcs_paths, _, gcs = augmented_models(GCS_ONLY_OUTPUT)
    if set(gcs["selected_rates_percent"]) != {"GCS"} or gcs != p11["gcs_training"]:
        raise ValueError("GCS models differ from the P11 evaluation")
    if frozen_hashes != from_json(EICU / "summary.json")["model_sha256"]:
        raise ValueError("The saved eICU predictions used different frozen models")
    if frozen_hashes != from_json(P9 / "summary.json")["model_sha256"]:
        raise ValueError("P9 used different frozen models")
    cohort, frames = eicu_inputs()
    frozen = pd.read_parquet(EICU / "predictions.parquet")
    gcs_scores, _, _ = score_frozen(frames, frames["hourly"], gcs_paths)
    if not frozen.index.equals(cohort.index):
        raise ValueError("Saved frozen eICU predictions differ from the common cohort")
    for row in p11["eicu"]:
        key = f"{row['mode']}_{row['model']}"
        for tag, scores in (("frozen", frozen[key].to_numpy()), ("gcs", gcs_scores[key])):
            observed = score_metrics(cohort.y.to_numpy(), scores)
            np.testing.assert_allclose(
                [observed[metric] for metric in METRICS],
                [row[f"{metric}_{tag}"] for metric in METRICS], rtol=0, atol=1e-12
            )
    features, _, _ = load_cohort()
    _, _, mask_hash = plan_for(features)
    reference_paths = [P11 / "summary.json", P11 / "set_c_three_way_replicates.csv",
                       P9 / "summary.json", P9 / "replicate_metrics.csv",
                       EICU / "summary.json", EICU / "cohort.parquet", EICU / "predictions.parquet"]
    reference_paths.extend(EICU / f"features_{mode}.parquet" for mode in MODES)
    reference_paths.extend(gcs_paths.values())
    sources = digests(reference_paths)
    with tempfile.TemporaryDirectory(dir=RUN, prefix="ma-gbt-refs-") as temporary:
        staging = Path(temporary) / "references"
        staging.mkdir()
        target = staging / "gcs_eicu_scores.parquet"
        pd.DataFrame(gcs_scores, index=cohort.index).to_parquet(target)
        meta = {"source_sha256": sources, "mask_draws_sha256": mask_hash, "gcs_scores_sha256": sha(target)}
        (staging / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        staging.rename(REFERENCES)
    print("Verified the original Set C and eICU comparison sources", flush=True)


def compare_c(pipes, reference, p11):
    features, y, _ = load_cohort()
    positions, plan, digest = plan_for(features)
    if digest != reference["mask_draws_sha256"]:
        raise ValueError("P9 mask draws changed between Python runtimes")
    scores, rhos = {}, {}
    for name, (pipe, columns) in pipes.items():
        scores[f"score_{name}"], rhos[name] = pipe.predict_and_rho(features.loc[:, columns])
    baseline, new = stress(features, y, pd.DataFrame(scores, index=features.index), pipes, positions, plan)
    old = pd.read_csv(P11 / "set_c_three_way_replicates.csv")
    frozen = pd.read_csv(P9 / "replicate_metrics.csv").rename(columns={metric: f"{metric}_p9" for metric in METRICS})
    checks = frozen.merge(old, on=KEYS, validate="one_to_one")
    for metric in METRICS:
        np.testing.assert_allclose(checks[f"{metric}_p9"], checks[f"{metric}_frozen"], rtol=0, atol=1e-12)
    prior_cols = [f"{metric}_{tag}" for metric in METRICS for tag in TAGS]
    new = new.rename(columns={metric: f"{metric}_ma" for metric in METRICS})
    paired = old[KEYS + prior_cols].merge(new, on=KEYS, validate="one_to_one")
    if not len(checks) == len(old) == len(frozen) == len(new) == len(paired) == 336:
        raise ValueError("Stress test replicate keys differ")
    for metric in METRICS:
        for tag in TAGS:
            paired[f"delta_{metric}_vs_{tag}"] = paired[f"{metric}_ma"] - paired[f"{metric}_{tag}"]

    clean = {}
    for row in p11["set_c"]:
        if row["scenario"] == "clean":
            name = row["model"]
            clean[name] = {**{col: row[col] for col in prior_cols}, "ma_rho": rhos[name]}
            for metric in METRICS:
                clean[name][f"{metric}_ma"] = baseline[name][metric]
                for tag in TAGS:
                    clean[name][f"delta_{metric}_vs_{tag}"] = baseline[name][metric] - row[f"{metric}_{tag}"]
    rows = [{"scenario": "clean", "model": name, "repeats": 1, **clean[name]} for name in MODELS]
    for (scenario, name), group in paired.groupby(["scenario", "model"], sort=False):
        row = {"scenario": scenario, "model": name, "repeats": len(group)}
        for metric in METRICS:
            for tag in (*TAGS, "ma"):
                column = f"{metric}_{tag}"
                row[column] = float(group[column].mean())
            for tag in TAGS:
                column = f"delta_{metric}_vs_{tag}"
                row[column] = float(group[column].mean())
                row[f"{column}_p10_p90"] = group[column].quantile([0.1, 0.9]).tolist()
            row[f"delta_{metric}_vs_clean_ma"] = row[f"{metric}_ma"] - clean[name][f"{metric}_ma"]
        rows.append(row)
    return rows, paired


def compare_eicu(pipes, reference, p11):
    cohort, frames = eicu_inputs()
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    if sha(gcs_path) != reference["gcs_scores_sha256"]:
        raise ValueError("GCS-only eICU predictions changed")
    sources = {"frozen": pd.read_parquet(EICU / "predictions.parquet"), "gcs": pd.read_parquet(gcs_path)}
    if not all(source.index.equals(cohort.index) for source in sources.values()):
        raise ValueError("Frozen eICU predictions do not align to the cohort")
    scores, rhos = {}, {}
    for mode in MODES:
        for name, (pipe, columns) in pipes.items():
            key = f"{mode}_{name}"
            scores[key], rhos[key] = pipe.predict_and_rho(frames[mode].loc[:, columns])
    print("Evaluating eICU hourly and native with the original patient bootstrap", flush=True)
    outcome = evaluate_predictions(cohort, scores)
    intervals = {
        tag: paired_intervals(cohort, {key: source[key].to_numpy() for key in scores}, scores)
        for tag, source in sources.items()
    }
    prior = {(row["mode"], row["model"]): row for row in p11["eicu"]}
    result = []
    for measured in outcome:
        mode, name = measured["mode"], measured["model"]
        key = f"{mode}_{name}"
        original = prior[mode, name]
        row = {"mode": mode, "model": name, "ma_rho": rhos[key], "ma_metrics": measured}
        for metric in METRICS:
            row[f"{metric}_ma"] = measured[metric]
            for tag in TAGS:
                value = original[f"{metric}_{tag}"]
                row[f"{metric}_{tag}"] = value
                row[f"delta_{metric}_vs_{tag}"] = measured[metric] - value
                if metric in ("ap", "auroc"):
                    row[f"delta_{metric}_vs_{tag}_ci"] = intervals[tag][key][metric]
        result.append(row)
    return result


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    from run_ma_gbt_p12 import validate_upstream

    validate_upstream()
    fit = check_fit()
    schemas = from_json(SCHEMA_FILE)["schemas"]
    reference = from_json(REFERENCES / "manifest.json")
    p11 = from_json(P11 / "summary.json")
    pipes = {}
    for name in MODELS:
        model = joblib.load(FITTED / f"{name}.joblib")
        schema = schemas[name]["final"]
        pipes[name] = (Predictor(model, schema), schema["columns"])
    set_c, masked = compare_c(pipes, reference, p11)
    eicu = compare_eicu(pipes, reference, p11)
    if any(sha(OUT / path) != value for path, value in reference["source_sha256"].items()):
        raise ValueError("A reference artifact changed")
    check_fit()
    report = {"training": fit, "set_c": set_c, "eicu": eicu, "reference": reference}
    with tempfile.TemporaryDirectory(dir=RUN, prefix="ma-gbt-eval-") as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        masked.to_csv(staging / "set_c_paired_replicates.csv", index=False)
        (staging / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(OUTPUT)
    print(f"MA-GBT evaluation saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-references", action="store_true", help="Use the existing XGBoost environment")
    if parser.parse_args().export_references:
        export_references()
    else:
        evaluate()
