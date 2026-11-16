"""Compare frozen augmented MA-GBT with original MA-GBT and GCS-only."""

import json
import tempfile
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE

import joblib
import numpy as np
import pandas as pd

from evaluate_eicu_demo import MODES, evaluate_predictions
from evaluate_ma_gbt_p12 import (
    FITTED as ORIGINAL_MODELS, KEYS, METRICS, OUTPUT as ORIGINAL_EVALUATION,
    P11, REFERENCES, Predictor, check_fit, compare_c, eicu_inputs,
)
from evaluate_missingness_augmentation import paired_intervals
from prepare import OUT, sha
from run_ma_gbt_p12 import validate_upstream
from stress_missingness_c import score_metrics
from train_ma_gbt_augmentation import OUTPUT as FITTED
from train_missingness_augmentation import MODELS, from_json

EVALUATION = FITTED / "evaluation"
TAGS = ("ma", "gcs")


def compare_set_c(pipes, reference, prior):
    current, new = compare_c(pipes, reference, from_json(P11 / "summary.json"))
    previous = pd.read_csv(ORIGINAL_EVALUATION / "set_c_paired_replicates.csv")
    columns = [f"{metric}_{tag}" for metric in METRICS for tag in TAGS]
    new = new.rename(columns={f"{metric}_ma": f"{metric}_aug" for metric in METRICS})
    merged = previous[KEYS + columns].merge(
        new[KEYS + [f"{metric}_{tag}" for metric in METRICS for tag in ("aug", "gcs")]],
        on=KEYS, suffixes=("", "_check"), validate="one_to_one",
    )
    if not len(merged) == len(previous) == len(new) == 336:
        raise ValueError("Expected all original Set C mask replicates")
    for metric in METRICS:
        name = f"{metric}_gcs"
        np.testing.assert_allclose(merged[name], merged.pop(f"{name}_check"), rtol=0, atol=1e-12)
        for tag in TAGS:
            merged[f"delta_{metric}_vs_{tag}"] = merged[f"{metric}_aug"] - merged[f"{metric}_{tag}"]

    old_rows = {(row["scenario"], row["model"]): row for row in prior["set_c"]}
    groups = {key: group for key, group in merged.groupby(["scenario", "model"], sort=False)}
    rows = []
    for row in current:
        key = row["scenario"], row["model"]
        old = old_rows[key]
        result = {**{col: row[col] for col in ("scenario", "model", "repeats")}, **{col: old[col] for col in columns}}
        if row["repeats"] != old["repeats"]:
            raise ValueError("Set C scenario replicate count changed")
        for metric in METRICS:
            result[f"{metric}_aug"] = row[f"{metric}_ma"]
            for tag in TAGS:
                col = f"delta_{metric}_vs_{tag}"
                result[col] = result[f"{metric}_aug"] - result[f"{metric}_{tag}"]
                if key in groups:
                    differences = groups[key][col]
                    np.testing.assert_allclose(differences.mean(), result[col], rtol=0, atol=1e-12)
                    result[f"{col}_p10_p90"] = differences.quantile([0.1, 0.9]).tolist()
            if "delta_ap_vs_clean_ma" in row:
                result[f"delta_{metric}_vs_clean_aug"] = row[f"delta_{metric}_vs_clean_ma"]
        if key[0] == "clean":
            result.update(rho_ma=old["ma_rho"], rho_aug=row["ma_rho"])
        rows.append(result)
    return rows, merged


def eicu_scores(pipes, frames):
    scores, rhos = {}, {}
    for mode in MODES:
        for name, (pipe, columns) in pipes.items():
            key = f"{mode}_{name}"
            scores[key], rhos[key] = pipe.predict_and_rho(frames[mode].loc[:, columns])
    return scores, rhos


def compare_eicu(new_pipes, original_pipes, reference, prior):
    cohort, frames = eicu_inputs()
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    if sha(gcs_path) != reference["gcs_scores_sha256"]:
        raise ValueError("Frozen GCS-only eICU predictions changed")
    gcs = pd.read_parquet(gcs_path)
    if not gcs.index.equals(cohort.index):
        raise ValueError("GCS-only eICU predictions do not align")
    baseline, original_rhos = eicu_scores(original_pipes, frames)
    prior_rows = {(row["mode"], row["model"]): row for row in prior["eicu"]}
    comparators = {"ma": baseline, "gcs": {key: gcs[key].to_numpy() for key in baseline}}
    scores, rhos = eicu_scores(new_pipes, frames)
    for key in scores:
        row = prior_rows[tuple(key.split("_"))]
        np.testing.assert_allclose(original_rhos[key], row["ma_rho"], rtol=0, atol=1e-12)
        for tag, probabilities in comparators.items():
            observed = score_metrics(cohort.y.to_numpy(), probabilities[key])
            np.testing.assert_allclose(
                [observed[m] for m in METRICS], [row[f"{m}_{tag}"] for m in METRICS], rtol=0, atol=1e-12
            )
    print("Evaluating both eICU cadences with paired patient-cluster bootstrap", flush=True)
    absolute = evaluate_predictions(cohort, scores)
    intervals = {tag: paired_intervals(cohort, source, scores) for tag, source in comparators.items()}
    result = []
    for values in absolute:
        mode, name = values["mode"], values["model"]
        key = f"{mode}_{name}"
        row = {
            "mode": mode, "model": name, "rho_ma": original_rhos[key], "rho_aug": rhos[key],
            "aug_metrics": values,
        }
        for metric in METRICS:
            row[f"{metric}_aug"] = values[metric]
            for tag in TAGS:
                old = prior_rows[mode, name][f"{metric}_{tag}"]
                row[f"{metric}_{tag}"] = old
                row[f"delta_{metric}_vs_{tag}"] = values[metric] - old
                if metric in intervals[tag][key]:
                    row[f"delta_{metric}_vs_{tag}_ci"] = intervals[tag][key][metric]
        result.append(row)
    return result


def pipelines(directory, schemas):
    return {
        name: (
            Predictor(joblib.load(directory / f"{name}.joblib"), schemas[name]["final"]),
            schemas[name]["final"]["columns"],
        )
        for name in MODELS
    }


def evaluate():
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    validate_upstream()
    manifest_path = FITTED / "manifest.json"
    fit = from_json(manifest_path)
    reference_path = REFERENCES / "manifest.json"
    reference = from_json(reference_path)
    prior_path = ORIGINAL_EVALUATION / "summary.json"
    prior = from_json(prior_path)
    original = check_fit()
    if prior["training"] != original or prior["reference"] != reference:
        raise ValueError("P13 evaluation and frozen model sources differ")
    if fit["training_dataset"] != "P12 A/B" or fit["device"] != "cpu":
        raise ValueError("Expected CPU models trained on A/B only")
    if fit["original_model_sha256"] != original["model_sha256"]:
        raise ValueError("P14 original MA-GBT model provenance differs from P13")
    sources = {
        **reference["source_sha256"],
        SCHEMA_FILE.relative_to(OUT).as_posix(): fit["encoded_schemas_sha256"],
        (ORIGINAL_MODELS / "manifest.json").relative_to(OUT).as_posix(): fit["original_manifest_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in (
            manifest_path, prior_path, reference_path, ORIGINAL_EVALUATION / "set_c_paired_replicates.csv"
        )},
        **{(FITTED / f"{name}.joblib").relative_to(OUT).as_posix(): fit["model_sha256"][name] for name in MODELS},
    }

    def verify():
        if any(sha(OUT / path) != digest for path, digest in sources.items()):
            raise ValueError("A model, training schema or reference source changed")
        check_fit()

    verify()
    schemas = from_json(SCHEMA_FILE)["schemas"]
    augmented = pipelines(FITTED, schemas)
    previous = pipelines(ORIGINAL_MODELS, schemas)
    set_c, replicates = compare_set_c(augmented, reference, prior)
    eicu = compare_eicu(augmented, previous, reference, prior)
    verify()
    report = {"training": fit, "set_c": set_c, "eicu": eicu, "source_sha256": sources}
    with tempfile.TemporaryDirectory(dir=FITTED, prefix="ma-gbt-aug-eval-") as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        replicates.to_csv(staging / "set_c_paired_replicates.csv", index=False)
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(EVALUATION)
    print(f"Augmented MA-GBT evaluation saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    evaluate()
