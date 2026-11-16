"""Decompose frozen eICU path reliance and measure a Set C bilirubin-only mask."""

import argparse
import hashlib
import json
import tempfile
import warnings
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE, encode

import numpy as np
import pandas as pd

from audit_ma_gbt_missingness import P16_SUMMARY, encoded_names, verify
from evaluate_ma_gbt_alpha10_urine import OUTPUT as URINE_FITTED
from evaluate_ma_gbt_alpha10_urine import xgb_missing_path_hits
from evaluate_ma_gbt_p12 import REFERENCES, eicu_inputs
from evaluate_missingness_augmentation import augmented_models
from prepare import OUT, sha
from stress_missingness_c import COVERAGE, METRICS, REPEATS, SEED, apply_masks, load_cohort, score_metrics
from train_missingness_augmentation import GCS_ONLY_OUTPUT, MODELS, from_json

LACTATE_FITTED = OUT / "ma_gbt_alpha10_lactate"
P18_SUMMARY = LACTATE_FITTED / "evaluation" / "summary.json"
OUTPUT = OUT / "ma_gbt_bilirubin_audit"
FIELD = "Bilirubin"


def partition(counter, schema):
    names = encoded_names(schema)
    bili = np.array([name.startswith("Bilirubin__") for name in names])
    lact = np.array([name.startswith("Lactate__") for name in names])
    any_count = counter(np.ones(len(names), dtype=bool))
    without_bili = counter(~bili)
    without_lactate = counter(~lact)
    without_both = counter(~(bili | lact))
    bili_count = counter(bili)
    lactate_count = counter(lact)
    return {
        "any": any_count, "bilirubin": bili_count, "lactate": lactate_count,
        "without_bilirubin": without_bili, "bilirubin_exclusive": any_count - without_bili,
        "other_than_bilirubin_lactate": without_both,
        "bilirubin_added_over_others": without_lactate - without_both,
        "lactate_added_after_bilirubin": any_count - without_lactate,
        "bilirubin_features": {
            name: counter(np.array([item == name for item in names]))
            for name in dict.fromkeys(name for name in names if name.startswith("Bilirubin__"))
        },
    }


def ma_eicu(pipe, frame):
    values, missing = encode(frame, pipe.schema)
    model = pipe.model
    active = {i for est in model.estimators_.ravel() for i in est.tree_.feature if i >= 0}
    active_mask = np.array([i in active for i in range(missing.shape[1])])

    def count(columns):
        selected = columns & active_mask
        if not missing[:, selected].any():
            return 0
        kept = np.zeros_like(missing)
        kept[:, selected] = missing[:, selected]
        return round(model.compute_missingness_reliance(values, kept) * len(frame))

    return partition(count, pipe.schema), model.predict_proba(values)[:, 1]


def gcs_eicu(pipeline, frame, schema):
    hits, scores = xgb_missing_path_hits(pipeline, frame, schema)

    def count(columns):
        return int(hits[:, columns].any(axis=1).sum())

    return partition(count, schema), scores


def views_set_c(frame):
    coverage = pd.read_csv(COVERAGE)
    entry = coverage.loc[coverage.cohort.eq("hourly") & coverage.variable.eq(FIELD)].iloc[0]
    observed = np.flatnonzero(frame[f"{FIELD}__count"].gt(0).to_numpy())
    if int(entry.n) <= 0 or not 0 < int(entry.observed_n) <= int(entry.n) or len(observed) == 0:
        raise ValueError("Unexpected original bilirubin coverage")
    target = (2 * int(entry.observed_n) * len(frame) + int(entry.n)) // (2 * int(entry.n))
    if len(observed) > target:
        partial_name = "matched_bilirubin"
        retained = target
    else:
        partial_name = "half_observed_bilirubin"
        retained = len(observed) // 2
    masked_n = len(observed) - retained
    plan = {FIELD: {"set_c_matched_observed": retained}}
    rng = np.random.default_rng(SEED)
    positions = [np.sort(rng.choice(observed, size=masked_n, replace=False)) for _ in range(REPEATS)]
    digest = hashlib.sha256(b"".join(rows.astype("<i8").tobytes() for rows in positions)).hexdigest()
    metadata = {
        "set_c_stays": len(frame), "set_c_observed": len(observed),
        "eicu_stays": int(entry.n), "eicu_observed": int(entry.observed_n), "eicu_coverage_target": target,
        "partial_design": partial_name, "partial_retained": retained, "partial_rows_to_mask": masked_n,
        "partial_repeats": REPEATS, "seed": SEED, "partial_mask_sha256": digest,
    }
    full = apply_masks(frame, {FIELD: observed}, plan, "full")
    other = [name for name in frame if not name.startswith(f"{FIELD}__")]
    pd.testing.assert_frame_equal(frame[other], full[other], check_exact=True)

    def views():
        yield "clean", 1, frame
        yield "full_bilirubin", 1, full
        for repeat, rows in enumerate(positions, start=1):
            yield partial_name, repeat, apply_masks(frame, {FIELD: rows}, plan, "matched")

    return metadata, views()


def score_c(frame, y, models, prior):
    metadata, views = views_set_c(frame)
    clean = {r["model"]: r for r in prior["set_c"] if r["scenario"] == "clean"}
    rows = []
    for scenario, repeat, view in views:
        for tag, group in models.items():
            for name, (pipe, columns) in group.items():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Found unknown categories.*")
                    probabilities = pipe.predict_proba(view.loc[:, columns])[:, 1]
                measures = score_metrics(y, probabilities)
                if scenario == "clean":
                    np.testing.assert_allclose(
                        [measures[m] for m in METRICS], [clean[name][f"{m}_{tag}"] for m in METRICS],
                        rtol=0, atol=1e-12,
                    )
                rows.append({
                    "scenario": scenario, "repeat": repeat, "tag": tag, "model": name,
                    **measures, "delta_ap": measures["ap"] - clean[name][f"ap_{tag}"],
                })
    return metadata, rows


def run(kind):
    destination = OUTPUT / f"{kind}.json"
    if destination.exists():
        raise FileExistsError(destination)
    p16, p18 = from_json(P16_SUMMARY), from_json(P18_SUMMARY)
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    ref = from_json(REFERENCES / "manifest.json")
    protected = {
        **p18["source_sha256"], **p16["source_sha256"],
        P18_SUMMARY.relative_to(OUT).as_posix(): sha(P18_SUMMARY),
        P16_SUMMARY.relative_to(OUT).as_posix(): sha(P16_SUMMARY),
        gcs_path.relative_to(OUT).as_posix(): ref["gcs_scores_sha256"],
    }
    verify(protected)
    schemas = from_json(SCHEMA_FILE)["schemas"]
    cohort, frames = eicu_inputs()
    original = {(row["mode"], row["model"]): row for row in p18["eicu"]}
    if kind == "gcs":
        _, gcs, manifest = augmented_models(GCS_ONLY_OUTPUT)
        if set(manifest["selected_rates_percent"]) != {"GCS"}:
            raise ValueError("Expected the frozen GCS-only models")
        scores = pd.read_parquet(gcs_path)
        if not scores.index.equals(cohort.index):
            raise ValueError("Frozen GCS-only scores do not align")
        models = {"gcs": gcs}
    else:
        from evaluate_ma_gbt_augmentation import pipelines
        from run_ma_gbt_p12 import validate_upstream

        validate_upstream()
        models = {"urine10": pipelines(URINE_FITTED, schemas), "lactate10": pipelines(LACTATE_FITTED, schemas)}

    eicu = []
    for mode, frame in frames.items():
        for tag, group in models.items():
            for name, (pipe, columns) in group.items():
                if list(columns) != schemas[name]["final"]["columns"]:
                    raise ValueError("A frozen schema does not match")
                if kind == "gcs":
                    counts, predicted = gcs_eicu(pipe, frame, schemas[name]["final"])
                    np.testing.assert_array_equal(predicted, scores[f"{mode}_{name}"].to_numpy(dtype=predicted.dtype))
                else:
                    counts, predicted = ma_eicu(pipe, frame.loc[:, columns])
                reference = original[mode, name]
                np.testing.assert_allclose(
                    np.array([counts["any"], counts["lactate"]]) / len(cohort),
                    [reference[f"rho_{tag}"], reference[f"lactate_rho_{tag}"]], rtol=0, atol=1e-12,
                )
                measures = score_metrics(cohort.y.to_numpy(), predicted)
                np.testing.assert_allclose(
                    [measures[m] for m in METRICS], [reference[f"{m}_{tag}"] for m in METRICS], rtol=0, atol=1e-12
                )
                eicu.append({"mode": mode, "tag": tag, "model": name, "stays": len(cohort), **counts})
    print(f"Verified {kind} eICU predictions and path partition", flush=True)
    frame, y, _ = load_cohort()
    plan, set_c = score_c(frame, y, models, p18)
    verify(protected)
    OUTPUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="bilirubin-audit-") as temporary:
        staging = Path(temporary) / destination.name
        staging.write_text(
            json.dumps({"device": "cpu", "eicu": eicu, "plan": plan, "set_c": set_c, "source_sha256": protected},
                       indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    print(f"Completed bilirubin-only Set C masks in {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("ma", "gcs"))
    run(parser.parse_args().model)
