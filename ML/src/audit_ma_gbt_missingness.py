"""Audit which fitted tree paths use imputed features, with no model fitting."""

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE, encode

import numpy as np
import pandas as pd

from evaluate_ma_gbt_alpha10_urine import EVALUATION as P16_EVALUATION
from evaluate_ma_gbt_alpha10_urine import OUTPUT as P16_FITTED
from evaluate_ma_gbt_alpha10_urine import xgb_missing_path_hits
from evaluate_ma_gbt_p12 import REFERENCES, eicu_inputs
from evaluate_missingness_augmentation import augmented_models
from prepare import OUT, sha
from stress_missingness_c import CHANNELS, COVERAGE, METRICS, PARTIAL, score_metrics
from train_missingness_augmentation import GCS_ONLY_OUTPUT, MODELS, from_json

OUTPUT = OUT / "ma_gbt_imputed_path_audit"
P16_SUMMARY = P16_EVALUATION / "summary.json"


def encoded_names(schema):
    return schema["numeric"] + [cat["name"] for cat in schema["categorical"] for _ in cat["levels"]]


def feature_groups(names, by_channel):
    groups = defaultdict(list)
    for index, name in enumerate(names):
        groups[name.split("__", 1)[0] if by_channel else name].append(index)
    return dict(groups)


def ma_path_audit(pipe, frame):
    values, mask = encode(frame, pipe.schema)
    names = encoded_names(pipe.schema)
    model = pipe.model
    scores = model.predict_proba(values)[:, 1]
    active = {i for tree in model.estimators_.ravel() for i in tree.tree_.feature if i >= 0}

    def counts(by_channel):
        result = {}
        for label, columns in feature_groups(names, by_channel).items():
            selected = [i for i in columns if i in active and mask[:, i].any()]
            if selected:
                subset = np.zeros_like(mask)
                subset[:, selected] = mask[:, selected]
                hits = model.compute_missingness_reliance(values, subset)
                result[label] = round(hits * len(frame))
            else:
                result[label] = 0
        return result

    return {
        "count_any": round(model.compute_missingness_reliance(values, mask) * len(frame)),
        "channels": counts(by_channel=True),
        "features": counts(by_channel=False),
    }, scores


def gcs_path_audit(pipeline, frame, schema):
    hits, scores = xgb_missing_path_hits(pipeline, frame, schema)
    names = encoded_names(schema)

    def counts(by_channel):
        return {name: int(hits[:, columns].any(axis=1).sum())
                for name, columns in feature_groups(names, by_channel).items()}

    return {
        "count_any": int(hits.any(axis=1).sum()),
        "channels": counts(by_channel=True),
        "features": counts(by_channel=False),
    }, scores


def verify(digests):
    for relative, digest in digests.items():
        if sha(OUT / relative) != digest:
            raise ValueError(f"Frozen source changed: {relative}")


def set_c_deltas(prior):
    prior_rows = {(row["scenario"], row["model"]): row for row in prior["set_c"]}
    rows = []
    # Keep matched and all-missing interventions separate because their severities differ.
    for design, channels in (("full", CHANNELS), ("matched", PARTIAL)):
        for channel in channels:
            for name in MODELS:
                clean = prior_rows["clean", name]
                perturbed = prior_rows[f"{design}_{channel}", name]
                row = {"design": design, "channel": channel, "model": name, "repeats": perturbed["repeats"]}
                for tag in ("urine10", "gcs"):
                    row[f"ap_clean_{tag}"] = clean[f"ap_{tag}"]
                    row[f"ap_masked_{tag}"] = perturbed[f"ap_{tag}"]
                    row[f"delta_ap_{tag}"] = perturbed[f"ap_{tag}"] - clean[f"ap_{tag}"]
                rows.append(row)
    return rows


def run(kind):
    destination = OUTPUT / f"{kind}.json"
    if destination.exists():
        raise FileExistsError(destination)
    prior = from_json(P16_SUMMARY)
    reference_path = REFERENCES / "manifest.json"
    reference = from_json(reference_path)
    gcs_scores_path = REFERENCES / "gcs_eicu_scores.parquet"
    sources = {
        **prior["source_sha256"], **reference["source_sha256"],
        P16_SUMMARY.relative_to(OUT).as_posix(): sha(P16_SUMMARY),
        gcs_scores_path.relative_to(OUT).as_posix(): reference["gcs_scores_sha256"],
    }
    verify(sources)
    cohort, frames = eicu_inputs()
    schemas = from_json(SCHEMA_FILE)["schemas"]
    coverage = pd.read_csv(COVERAGE)
    frozen_rows = {(row["mode"], row["model"]): row for row in prior["eicu"]}
    if kind == "gcs":
        _, pipes, manifest = augmented_models(GCS_ONLY_OUTPUT)
        if set(manifest["selected_rates_percent"]) != {"GCS"}:
            raise ValueError("Expected the frozen GCS-only models")
        stored = pd.read_parquet(gcs_scores_path)
        if not stored.index.equals(cohort.index):
            raise ValueError("Frozen GCS-only scores differ in row order")
    else:
        from evaluate_ma_gbt_augmentation import pipelines
        from run_ma_gbt_p12 import validate_upstream

        validate_upstream()
        pipes = pipelines(P16_FITTED, schemas)

    result = []
    for mode, frame in frames.items():
        observed = coverage.loc[coverage.cohort.eq(mode)].set_index("variable")["observed_n"].to_dict()
        for name in MODELS:
            pipe, columns = pipes[name]
            schema = schemas[name]["final"]
            if list(columns) != schema["columns"]:
                raise ValueError("Frozen columns differ from the encoded schema")
            if kind == "gcs":
                audit, scores = gcs_path_audit(pipe, frame, schema)
                np.testing.assert_array_equal(scores, stored[f"{mode}_{name}"].to_numpy(dtype=scores.dtype))
                tag = "gcs"
            else:
                audit, scores = ma_path_audit(pipe, frame.loc[:, columns])
                tag = "urine10"
                np.testing.assert_allclose(
                    audit["count_any"] / len(frame), frozen_rows[mode, name]["rho_urine10"], rtol=0, atol=1e-12
                )
            expected = frozen_rows[mode, name]
            metrics = score_metrics(cohort.y.to_numpy(), scores)
            np.testing.assert_allclose(
                [metrics[m] for m in METRICS], [expected[f"{m}_{tag}"] for m in METRICS], rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                audit["channels"].get("Urine", 0) / len(frame), expected[f"urine_rho_{tag}"], rtol=0, atol=1e-12
            )
            if any(count > audit["count_any"] for count in audit["channels"].values()):
                raise ValueError("A channel exceeds the cohort total")
            result.append({"mode": mode, "model": name, "stays": len(frame), "observed": observed, **audit})
            print(f"Verified {kind} {mode} {name} predictions and path counts", flush=True)

    verify(sources)
    OUTPUT.mkdir(exist_ok=True)
    payload = {
        "device": "cpu", "p16_summary_sha256": sources[P16_SUMMARY.relative_to(OUT).as_posix()],
        "eicu": result, "set_c": set_c_deltas(prior), "source_sha256": sources,
    }
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="path-audit-") as temporary:
        staged = Path(temporary) / destination.name
        staged.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(destination)
    print(f"Path audit saved to {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("ma", "gcs"))
    run(parser.parse_args().model)
