"""Compare GCS-only, GCS-plus-urine, and original predictors on fixed holdouts."""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_eicu_demo import MODES, score_frozen
from evaluate_missingness_augmentation import (
    DEFAULT_OUTPUT as DUAL_EVALUATION, EICU, METRICS,
    augmented_models, paired_intervals, read_json,
)
from prepare import OUT, sha
from stress_missingness_c import score_metrics
from train_missingness_augmentation import DEFAULT_OUTPUT as DUAL_MODELS, GCS_ONLY_OUTPUT


GCS_EVALUATION = OUT / "gcs_only_augmentation_evaluation"
DEFAULT_OUTPUT = OUT / "gcs_only_comparison"


def checked_run(models, evaluation, channels):
    paths, _, manifest = augmented_models(models)
    result = read_json(evaluation / "summary.json")
    if set(manifest["selected_rates_percent"]) != set(channels) or result["training"] != manifest:
        raise ValueError("Evaluation does not belong to the expected augmentation model files")
    return paths, result


def triple_metrics(dual, gcs):
    values = {}
    for metric in METRICS:
        baseline = gcs[f"{metric}_frozen"]
        if abs(dual[f"{metric}_frozen"] - baseline) > 1e-12:
            raise ValueError("Original-model baseline differs between evaluations")
        old_aug = dual[f"{metric}_augmented"]
        new_aug = gcs[f"{metric}_augmented"]
        values.update({
            f"{metric}_frozen": baseline, f"{metric}_dual": old_aug, f"{metric}_gcs": new_aug,
            f"delta_{metric}_vs_frozen": new_aug - baseline,
            f"delta_{metric}_vs_dual": new_aug - old_aug,
        })
    return values


def compare_c(dual, gcs, gcs_evaluation):
    dual_repeats = pd.read_csv(DUAL_EVALUATION / "set_c_paired_replicates.csv")
    gcs_repeats = pd.read_csv(gcs_evaluation / "set_c_paired_replicates.csv")
    keys = ["scenario", "model", "repeat", "design", "variables"]
    pairs = dual_repeats.merge(gcs_repeats, on=keys, suffixes=("_dual", "_gcs"), validate="one_to_one")
    if not len(pairs) == len(dual_repeats) == len(gcs_repeats) == 336:
        raise ValueError("The two runs do not have all the same P9 replicates")
    paired = pairs.loc[:, keys].copy()
    for metric in METRICS:
        np.testing.assert_allclose(pairs[f"{metric}_frozen_dual"], pairs[f"{metric}_frozen_gcs"], rtol=0, atol=1e-12)
        paired[f"{metric}_frozen"] = pairs[f"{metric}_frozen_gcs"]
        paired[f"{metric}_dual"] = pairs[f"{metric}_augmented_dual"]
        paired[f"{metric}_gcs"] = pairs[f"{metric}_augmented_gcs"]
        paired[f"delta_{metric}_vs_dual"] = paired[f"{metric}_gcs"] - paired[f"{metric}_dual"]

    prior = {(row["scenario"], row["model"]): row for row in dual["set_c"]}
    output = []
    for row in gcs["set_c"]:
        key = (row["scenario"], row["model"])
        result = {"scenario": key[0], "model": key[1], "repeats": row["repeats"], **triple_metrics(prior[key], row)}
        if key[0] != "clean":
            matches = paired.loc[paired.scenario.eq(key[0]) & paired.model.eq(key[1])]
            if len(matches) != row["repeats"]:
                raise ValueError("Unexpected number of matched draws")
            for metric in METRICS:
                change = matches[f"delta_{metric}_vs_dual"]
                np.testing.assert_allclose(change.mean(), result[f"delta_{metric}_vs_dual"], rtol=0, atol=1e-12)
                result[f"delta_{metric}_vs_dual_p10_p90"] = change.quantile([0.1, 0.9]).tolist()
        output.append(result)
    if len(output) != len(prior) or len(output) != 28:
        raise ValueError("Missing Set C scenarios")
    return output, paired


def compare_eicu(dual, gcs, dual_models, gcs_models):
    cohort = pd.read_parquet(EICU / "cohort.parquet")
    frames = {mode: pd.read_parquet(EICU / f"features_{mode}.parquet") for mode in MODES}
    if not cohort.index.is_unique or not all(frame.index.equals(cohort.index) for frame in frames.values()):
        raise ValueError("eICU inputs differ from the common cohort")
    y = cohort.y.to_numpy()
    old, _, _ = score_frozen(frames, frames["hourly"], dual_models)
    new, _, _ = score_frozen(frames, frames["hourly"], gcs_models)
    prior = {(row["mode"], row["model"]): row for row in dual["eicu"]}
    comparisons = paired_intervals(cohort, old, new)
    if len(prior) != len(gcs["eicu"]) or len(gcs["eicu"]) != 4:
        raise ValueError("Missing eICU model or cadence")
    output = []
    for row in gcs["eicu"]:
        key = (row["mode"], row["model"])
        score_key = "_".join(key)
        result = {"mode": key[0], "model": key[1], **triple_metrics(prior[key], row)}
        for tag, scores in (("dual", old), ("gcs", new)):
            values = score_metrics(y, scores[score_key])
            np.testing.assert_allclose([values[m] for m in METRICS], [result[f"{m}_{tag}"] for m in METRICS], atol=1e-12, rtol=0)
        for metric in ("ap", "auroc"):
            result[f"delta_{metric}_vs_frozen_ci"] = row[f"delta_{metric}_paired_ci"]
            result[f"delta_{metric}_vs_dual_ci"] = comparisons[score_key][metric]
        result["paired_bootstraps"] = comparisons[score_key]["n"]
        output.append(result)
    return output


def run(models, evaluation, output):
    output = output.resolve()
    root = OUT.resolve()
    if not output.is_relative_to(root) or output == root or output.exists() or not output.parent.is_dir():
        raise ValueError("Use a new output directory under ML/artifacts")
    dual_paths, dual = checked_run(DUAL_MODELS, DUAL_EVALUATION, ("GCS", "Urine"))
    gcs_paths, gcs = checked_run(models, evaluation, ("GCS",))
    if dual["training"]["original_model_sha256"] != gcs["training"]["original_model_sha256"]:
        raise ValueError("The original-model references differ")
    for name in ("features_AB.parquet", "labels_AB.csv", "folds3.csv"):
        if dual["training"]["source_sha256"][name] != gcs["training"]["source_sha256"][name]:
            raise ValueError(f"The two training runs used different A/B inputs: {name}")
    immutable = [*dual_paths.values(), *gcs_paths.values(), DUAL_MODELS / "manifest.json", models / "manifest.json",
                 EICU / "cohort.parquet", *(EICU / f"features_{mode}.parquet" for mode in MODES)]
    immutable += [folder / file for folder in (DUAL_EVALUATION, evaluation)
                  for file in ("summary.json", "set_c_paired_replicates.csv")]
    hashes = {path: sha(path) for path in immutable}
    print("Checking all three versions on the identical Set C masks", flush=True)
    set_c, paired = compare_c(dual, gcs, evaluation)
    print("Comparing both augmentation versions on the same eICU patients", flush=True)
    with threadpool_limits(limits=2):
        eicu = compare_eicu(dual, gcs, dual_paths, gcs_paths)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("A comparison input changed")
    summary = {
        "gcs_training": gcs["training"], "comparison_device": "cpu", "comparison_fits": 0,
        "set_c": set_c, "eicu": eicu,
    }
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="gcs-comparison-") as temporary:
        staged = Path(temporary).resolve() / "run"
        if staged.parent.parent != output.parent:
            raise RuntimeError("Unexpected temporary path")
        staged.mkdir()
        pd.DataFrame(set_c).to_csv(staged / "set_c_three_way.csv", index=False)
        paired.to_csv(staged / "set_c_three_way_replicates.csv", index=False)
        pd.DataFrame(eicu).to_csv(staged / "eicu_three_way.csv", index=False)
        (staged / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(output)
    print(f"Saved three-way aggregate comparisons under {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=GCS_ONLY_OUTPUT)
    parser.add_argument("--evaluation", type=Path, default=GCS_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.models, args.evaluation, args.output)
