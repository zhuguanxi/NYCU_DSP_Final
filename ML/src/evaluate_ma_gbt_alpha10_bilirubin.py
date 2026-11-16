"""Compare frozen bilirubin augmentation on the original and P19 masking designs."""

import json
import tempfile
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE

import numpy as np
import pandas as pd

from audit_bilirubin_reliance import OUTPUT as P19, ma_eicu, score_c
from evaluate_eicu_demo import evaluate_predictions
from evaluate_ma_gbt_alpha10_lactate import EVALUATION as P18_EVALUATION
from evaluate_ma_gbt_alpha10_urine import paired_set_c
from evaluate_ma_gbt_augmentation import eicu_scores, pipelines, reference_metrics
from evaluate_ma_gbt_p12 import METRICS, REFERENCES, eicu_inputs
from evaluate_missingness_augmentation import paired_intervals
from prepare import OUT, sha
from run_ma_gbt_p12 import validate_upstream
from stress_missingness_c import load_cohort
from train_ma_gbt_alpha10 import ALPHA, verify_hashes
from train_ma_gbt_alpha10_bilirubin import CANDIDATES, FIXED_RATES, OUTPUT, PRIMARY, sources_for_p18
from train_ma_gbt_alpha10_lactate import OUTPUT as P18_FITTED
from train_missingness_augmentation import MODELS, from_json

NEW = "bilirubin10"
BASELINES = ("lactate10", "gcs")
EVALUATION = OUTPUT / "evaluation"


def compare_p19(new, set_c, audit):
    frame, y, _ = load_cohort()
    plan, new_rows = score_c(frame, y, {NEW: new}, {"set_c": set_c})
    if plan != audit["ma"]["plan"] or plan != audit["gcs"]["plan"]:
        raise ValueError("P19 bilirubin mask positions differ")
    controls = [
        row for group in audit.values() for row in group["set_c"] if row["tag"] in BASELINES
    ]
    columns = ["scenario", "repeat", "model"]
    measure_columns = [*METRICS, "delta_ap"]
    paired = pd.DataFrame(controls + new_rows).pivot(index=columns, columns="tag", values=measure_columns)
    if len(paired) != 2 * (2 + plan["partial_repeats"]) or paired.isna().any().any():
        raise ValueError("Incomplete P19 bilirubin-only comparison")
    paired.columns = [f"{metric}_{tag}" for metric, tag in paired.columns]
    paired = paired.reset_index()
    for tag in BASELINES:
        paired[f"delta_ap_vs_{tag}"] = paired[f"ap_{NEW}"] - paired[f"ap_{tag}"]
    summary = []
    for (scenario, name), group in paired.groupby(["scenario", "model"], sort=False):
        row = {"scenario": scenario, "model": name, "repeats": len(group)}
        row.update(group.drop(columns=columns).mean().to_dict())
        for tag in BASELINES:
            column = f"delta_ap_vs_{tag}"
            row[f"{column}_p10_p90"] = group[column].quantile([0.1, 0.9]).tolist()
        summary.append(row)
    return plan, summary, paired


def compare_eicu(new, old, reference, prior, audits):
    cohort, frames = eicu_inputs()
    targets = cohort.y.to_numpy()
    old_scores, old_rho = eicu_scores(old, frames)
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    if sha(gcs_path) != reference["gcs_scores_sha256"]:
        raise ValueError("The frozen GCS-only scores changed")
    gcs = pd.read_parquet(gcs_path)
    if not gcs.index.equals(cohort.index):
        raise ValueError("GCS-only scores differ in row order")
    controls = {"lactate10": old_scores, "gcs": {key: gcs[key].to_numpy() for key in old_scores}}
    frozen = {(row["mode"], row["model"]): row for row in prior["eicu"]}
    audited = {
        (row["tag"], row["mode"], row["model"]): row
        for path in audits.values() for row in path["eicu"] if row["tag"] in BASELINES
    }
    known_bilirubin = {
        (row["mode"], row["model"]): row["channels"]["Bilirubin"] for row in prior["new_path_audit"]
    }
    new_scores, new_paths = {}, {}
    for mode, frame in frames.items():
        for name, (pipe, columns) in new.items():
            key = f"{mode}_{name}"
            new_paths[mode, name], new_scores[key] = ma_eicu(pipe, frame.loc[:, columns])
            for tag, scores in controls.items():
                stats = reference_metrics(targets, scores[key])
                expected = frozen[mode, name]
                np.testing.assert_allclose(
                    [stats[m] for m in METRICS], [expected[f"{m}_{tag}"] for m in METRICS],
                    rtol=0, atol=1e-12,
                )
                old_path = audited[tag, mode, name]
                if old_path["stays"] != len(frame):
                    raise ValueError("P19 path counts used a different cohort")
                np.testing.assert_allclose(old_path["any"] / len(frame), expected[f"rho_{tag}"], rtol=0, atol=1e-12)
            before = audited["lactate10", mode, name]
            if before["bilirubin"] != known_bilirubin[mode, name]:
                raise ValueError("P18 and P19 bilirubin path counts differ")
            np.testing.assert_allclose(before["any"] / len(frame), old_rho[key], rtol=0, atol=1e-12)

    print("Evaluating the original eICU cadences and paired patient intervals", flush=True)
    absolute = evaluate_predictions(cohort, new_scores)
    contrasts = {tag: paired_intervals(cohort, scores, new_scores) for tag, scores in controls.items()}
    rows = []
    for measured in absolute:
        mode, name = measured["mode"], measured["model"]
        key = f"{mode}_{name}"
        row = {"mode": mode, "model": name, "new_metrics": measured}
        paths = {NEW: new_paths[mode, name], **{tag: audited[tag, mode, name] for tag in BASELINES}}
        for tag, info in paths.items():
            row[f"rho_{tag}"] = info["any"] / len(cohort)
            row[f"bilirubin_rho_{tag}"] = info["bilirubin"] / len(cohort)
            row[f"bilirubin_exclusive_rho_{tag}"] = info["bilirubin_exclusive"] / len(cohort)
        for metric in METRICS:
            row[f"{metric}_{NEW}"] = measured[metric]
            for tag in BASELINES:
                baseline = frozen[mode, name][f"{metric}_{tag}"]
                row[f"{metric}_{tag}"] = baseline
                row[f"delta_{metric}_vs_{tag}"] = measured[metric] - baseline
                if metric in contrasts[tag][key]:
                    row[f"delta_{metric}_vs_{tag}_ci"] = contrasts[tag][key][metric]
        rows.append(row)
    return rows, [{"mode": mode, "model": name, **vals} for (mode, name), vals in new_paths.items()]


def evaluate():
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    validate_upstream()
    fit_path = OUTPUT / "manifest.json"
    fit = from_json(fit_path)
    previous, sources = sources_for_p18()
    if (
        fit["alpha"] != ALPHA or fit["fixed_rates_percent"] != FIXED_RATES or fit["source_sha256"] != sources
        or fit["primary_model"] != PRIMARY or fit["selected_bilirubin_rate_percent"] not in CANDIDATES
        or (fit["training_dataset"], fit["device"]) != ("P12 A/B", "cpu")
    ):
        raise ValueError("Expected the frozen P20 A/B models")
    reference_path = REFERENCES / "manifest.json"
    prior_path = P18_EVALUATION / "summary.json"
    replicate_path = P18_EVALUATION / "set_c_paired_replicates.csv"
    audits_paths = {tag: P19 / f"{tag}.json" for tag in ("ma", "gcs")}
    reference = from_json(reference_path)
    prior = from_json(prior_path)
    if prior["training"] != previous:
        raise ValueError("P18 evaluation does not belong to the frozen models")
    audits = {tag: from_json(path) for tag, path in audits_paths.items()}
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    extra = (fit_path, prior_path, reference_path, replicate_path, *audits_paths.values())
    protected = {
        **sources, **prior["source_sha256"], **reference["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in extra},
        **{(OUTPUT / f"{name}.joblib").relative_to(OUT).as_posix(): fit["model_sha256"][name] for name in MODELS},
        gcs_path.relative_to(OUT).as_posix(): reference["gcs_scores_sha256"],
    }
    for audit in audits.values():
        if audit["device"] != "cpu":
            raise ValueError("Expected the frozen CPU path audit")
        for relative, digest in audit["source_sha256"].items():
            if relative in protected and protected[relative] != digest:
                raise ValueError(f"P19 source hash differs: {relative}")
            protected[relative] = digest
    verify_hashes(protected)
    schemas = from_json(SCHEMA_FILE)["schemas"]
    new = pipelines(OUTPUT, schemas)
    old = pipelines(P18_FITTED, schemas)
    set_c, replicates, _ = paired_set_c(new, reference, prior, replicate_path, BASELINES, NEW)
    plan, bili_rows, bili_pairs = compare_p19(new, set_c, audits)
    eicu, paths = compare_eicu(new, old, reference, prior, audits)
    verify_hashes(protected)
    report = {
        "training": fit, "set_c": set_c, "set_c_bilirubin": bili_rows, "bilirubin_mask_plan": plan,
        "eicu": eicu, "new_path_audit": paths, "source_sha256": protected,
    }
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="ma-gbt-bilirubin-eval-") as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        replicates.to_csv(staging / "set_c_paired_replicates.csv", index=False)
        bili_pairs.to_csv(staging / "set_c_bilirubin_paired_replicates.csv", index=False)
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(EVALUATION)
    print(f"Bilirubin augmentation comparison saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    evaluate()
