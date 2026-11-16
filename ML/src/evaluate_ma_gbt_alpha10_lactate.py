"""Evaluate the lactate augmentation on the existing Set C and eICU cohorts."""

import json
import tempfile
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE

import numpy as np
import pandas as pd

from audit_ma_gbt_missingness import OUTPUT as P17, ma_path_audit
from evaluate_eicu_demo import evaluate_predictions
from evaluate_ma_gbt_alpha10_urine import EVALUATION as P16_EVALUATION
from evaluate_ma_gbt_alpha10_urine import paired_set_c
from evaluate_ma_gbt_augmentation import eicu_scores, pipelines, reference_metrics
from evaluate_ma_gbt_p12 import METRICS, REFERENCES, eicu_inputs
from evaluate_missingness_augmentation import paired_intervals
from prepare import OUT, sha
from run_ma_gbt_p12 import validate_upstream
from train_ma_gbt_alpha10 import ALPHA, verify_hashes
from train_ma_gbt_alpha10_lactate import CANDIDATES, FIXED_RATES, OUTPUT, sources_for_p16
from train_ma_gbt_alpha10_urine import OUTPUT as P16_FITTED
from train_missingness_augmentation import MODELS, from_json

NEW = "lactate10"
BASELINES = ("urine10", "gcs")
CHANNELS = ("Lactate", "Urine", "GCS")
EVALUATION = OUTPUT / "evaluation"


def compare_eicu(new, old, reference, prior, audits):
    cohort, frames = eicu_inputs()
    y = cohort.y.to_numpy()
    old_scores, old_rho = eicu_scores(old, frames)
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    if sha(gcs_path) != reference["gcs_scores_sha256"]:
        raise ValueError("The frozen GCS-only scores changed")
    gcs = pd.read_parquet(gcs_path)
    if not gcs.index.equals(cohort.index):
        raise ValueError("GCS-only scores differ in row order")
    controls = {"urine10": old_scores, "gcs": {key: gcs[key].to_numpy() for key in old_scores}}
    frozen = {(r["mode"], r["model"]): r for r in prior["eicu"]}
    p17_paths = {tag: {(r["mode"], r["model"]): r for r in audit["eicu"]} for tag, audit in audits.items()}
    new_scores, new_paths = {}, {}
    for mode, frame in frames.items():
        for name, (pipe, columns) in new.items():
            key = f"{mode}_{name}"
            new_paths[mode, name], new_scores[key] = ma_path_audit(pipe, frame.loc[:, columns])
            for tag, scores in controls.items():
                old_stats = reference_metrics(y, scores[key])
                np.testing.assert_allclose(
                    [old_stats[m] for m in METRICS], [frozen[mode, name][f"{m}_{tag}"] for m in METRICS],
                    rtol=0, atol=1e-12,
                )
                audited = p17_paths[tag][mode, name]
                if audited["stays"] != len(frame):
                    raise ValueError("The audit used a different eICU cohort")
                np.testing.assert_allclose(
                    audited["channels"]["Urine"] / len(frame), frozen[mode, name][f"urine_rho_{tag}"], rtol=0, atol=1e-12
                )
            np.testing.assert_allclose(old_rho[key], p17_paths["urine10"][mode, name]["count_any"] / len(frame), rtol=0, atol=1e-12)

    print("Evaluating both eICU cadences with the existing patient bootstrap", flush=True)
    absolute = evaluate_predictions(cohort, new_scores)
    contrasts = {tag: paired_intervals(cohort, scores, new_scores) for tag, scores in controls.items()}
    rows = []
    for measured in absolute:
        mode, name = measured["mode"], measured["model"]
        key = f"{mode}_{name}"
        row = {"mode": mode, "model": name, "new_metrics": measured}
        paths = {NEW: new_paths[mode, name], **{tag: p17_paths[tag][mode, name] for tag in BASELINES}}
        for tag, info in paths.items():
            row[f"rho_{tag}"] = info["count_any"] / len(cohort)
            for channel in CHANNELS:
                row[f"{channel.lower()}_rho_{tag}"] = info["channels"].get(channel, 0) / len(cohort)
        for metric in METRICS:
            row[f"{metric}_{NEW}"] = measured[metric]
            for tag in BASELINES:
                old_measure = frozen[mode, name][f"{metric}_{tag}"]
                row[f"{metric}_{tag}"] = old_measure
                row[f"delta_{metric}_vs_{tag}"] = measured[metric] - old_measure
                if metric in contrasts[tag][key]:
                    row[f"delta_{metric}_vs_{tag}_ci"] = contrasts[tag][key][metric]
        rows.append(row)
    return rows, list(
        {"mode": mode, "model": name, **values} for (mode, name), values in new_paths.items()
    )


def evaluate():
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    validate_upstream()
    fit_path = OUTPUT / "manifest.json"
    fit = from_json(fit_path)
    previous, sources = sources_for_p16()
    if (
        fit["alpha"] != ALPHA or fit["fixed_rates_percent"] != FIXED_RATES or fit["source_sha256"] != sources
        or fit["selected_lactate_rate_percent"] not in CANDIDATES
        or (fit["training_dataset"], fit["device"]) != ("P12 A/B", "cpu")
    ):
        raise ValueError("Expected frozen CPU models with the fixed P18 training protocol")
    reference_path = REFERENCES / "manifest.json"
    prior_path = P16_EVALUATION / "summary.json"
    replicate_path = P16_EVALUATION / "set_c_paired_replicates.csv"
    audits_paths = {"urine10": P17 / "ma.json", "gcs": P17 / "gcs.json"}
    reference = from_json(reference_path)
    prior = from_json(prior_path)
    if prior["training"] != previous:
        raise ValueError("P16 evaluation and frozen training manifest differ")
    audits = {tag: from_json(path) for tag, path in audits_paths.items()}
    prior_hash = sha(prior_path)
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    extra = (fit_path, prior_path, reference_path, replicate_path, *audits_paths.values())
    protected = {
        **sources, **prior["source_sha256"], **reference["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in extra},
        **{(OUTPUT / f"{name}.joblib").relative_to(OUT).as_posix(): fit["model_sha256"][name] for name in MODELS},
        gcs_path.relative_to(OUT).as_posix(): reference["gcs_scores_sha256"],
    }
    for audit in audits.values():
        if audit["p16_summary_sha256"] != prior_hash:
            raise ValueError("P17 path audit used a different P16 evaluation")
        for relative, digest in audit["source_sha256"].items():
            if relative in protected and protected[relative] != digest:
                raise ValueError(f"P17 audit source hash differs: {relative}")
            protected[relative] = digest
    verify_hashes(protected)
    schemas = from_json(SCHEMA_FILE)["schemas"]
    new = pipelines(OUTPUT, schemas)
    old = pipelines(P16_FITTED, schemas)
    set_c, replicates, _ = paired_set_c(new, reference, prior, replicate_path, BASELINES, NEW)
    eicu, paths = compare_eicu(new, old, reference, prior, audits)
    verify_hashes(protected)
    report = {
        "training": fit, "set_c": set_c, "eicu": eicu, "new_path_audit": paths, "source_sha256": protected,
    }
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="ma-gbt-lactate-eval-") as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        replicates.to_csv(staging / "set_c_paired_replicates.csv", index=False)
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(EVALUATION)
    print(f"Lactate augmentation comparison saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    evaluate()
