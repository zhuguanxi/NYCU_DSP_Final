"""Choose lactate masking on the original A/B folds and freeze both MA-GBT models."""

import argparse
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE, encode

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from prepare import OUT, sha
from run_ma_gbt_p12 import PARAMETERS, UPSTREAM_SHA, validate_upstream
from train_ma_gbt_alpha10 import ALPHA, RATES, verify_hashes
from train_ma_gbt_alpha10_urine import OUTPUT as P16_FITTED
from train_ma_gbt_augmentation import (
    MASK_SEED, VALIDATION_PARTIAL, VALIDATION_REPEATS, fit_one, mask_features,
)
from train_missingness_augmentation import FOLD_IDS, MODELS, ab_inputs, from_json

OUTPUT = OUT / "ma_gbt_alpha10_lactate"
FIXED_RATES = {**RATES, "Urine": 75}
CANDIDATES = (25, 75, 100)
AP_TOLERANCE = 0.01
VALIDATION_JOINT = {**VALIDATION_PARTIAL, "Urine": 100, "Lactate": 75}
METRICS = ("ap", "auroc", "lactate_rho")
NO_MASKS = dict.fromkeys(FIXED_RATES, 0)


def validation_scores(model, frame, target, schema, fold):
    slots = [i for i, name in enumerate(schema["numeric"]) if name.startswith("Lactate__")]

    def score(view):
        features, mask = encode(view, schema)
        prediction = model.predict_proba(features)[:, 1]
        lactate = np.zeros_like(mask)
        lactate[:, slots] = mask[:, slots]
        return [
            float(average_precision_score(target, prediction)),
            float(roc_auc_score(target, prediction)),
            float(model.compute_missingness_reliance(features, lactate)),
        ]

    rows = {"clean": score(frame)}
    for name, rates in (
        ("full_lactate", {**NO_MASKS, "Lactate": 100}),
        ("full_gcs", {**NO_MASKS, "GCS": 100, "Lactate": 0}),
    ):
        view, _ = mask_features(frame, rates, np.random.SeedSequence([MASK_SEED, fold, 2]))
        rows[name] = score(view)
    joint = [
        score(mask_features(frame, VALIDATION_JOINT, np.random.SeedSequence([MASK_SEED, fold, 1, i]))[0])
        for i in range(VALIDATION_REPEATS)
    ]
    rows["joint"] = np.mean(joint, axis=0).tolist()
    return {name: dict(zip(METRICS, values, strict=True)) for name, values in rows.items()}


def sources_for_p16():
    path = P16_FITTED / "manifest.json"
    saved = from_json(path)
    if (
        saved["alpha"] != ALPHA or saved["parameters"] != PARAMETERS
        or saved["upstream_commit"] != UPSTREAM_SHA or saved["fixed_rates_percent"] != RATES
        or saved["selected_urine_rate_percent"] != FIXED_RATES["Urine"] or saved["mask_seed"] != MASK_SEED
        or (saved["training_dataset"], saved["device"]) != ("P12 A/B", "cpu")
    ):
        raise ValueError("P16 did not use the expected frozen A/B setup")
    sources = {
        **saved["source_sha256"],
        path.relative_to(OUT).as_posix(): sha(path),
        **{(P16_FITTED / f"{name}.joblib").relative_to(OUT).as_posix(): saved["model_sha256"][name] for name in MODELS},
    }
    verify_hashes(sources)
    return saved, sources


def check_masks():
    frame, _, _ = ab_inputs()
    probe = frame.iloc[:300]
    seed = np.random.SeedSequence([MASK_SEED, 0, 0])
    original, counts = mask_features(probe, FIXED_RATES, seed)
    masked, added = mask_features(probe, {**FIXED_RATES, "Lactate": 100}, seed)
    other = [name for name in probe if not name.startswith("Lactate__")]
    pd.testing.assert_frame_equal(original[other], masked[other], check_exact=True)
    if {key: added[key] for key in FIXED_RATES} != counts:
        raise ValueError("Adding lactate changed the original training masks")
    if not (masked["Lactate__count"].eq(0).all() and masked["Lactate__missing"].eq(1).all()):
        raise ValueError("Lactate all-missing masking failed")
    print("A/B masking probe leaves all non-lactate fields identical to P16", flush=True)


def aggregate(rows):
    means = {
        view: {metric: float(np.mean([row["views"][view][metric] for row in rows])) for metric in METRICS}
        for view in rows[0]["views"]
    }
    return {
        "views": means,
        "mean_masked_ap": (means["full_lactate"]["ap"] + means["joint"]["ap"]) / 2,
        "mean_masked_lactate_rho": (means["full_lactate"]["lactate_rho"] + means["joint"]["lactate_rho"]) / 2,
    }


def fit(schemas, name, fold, strength):
    return fit_one(
        schemas, name, fold, str(strength), ALPHA,
        rates={**FIXED_RATES, "Lactate": strength}, validation_fn=validation_scores,
    )


def train(workers):
    validate_upstream()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    previous, sources = sources_for_p16()
    schemas = from_json(SCHEMA_FILE)["schemas"]
    reference = previous["candidates_full"][str(FIXED_RATES["Urine"])]["views"]
    guard_views = ("clean", "full_gcs")
    reference_ap = {key: reference[key]["ap"] for key in guard_views}
    check_masks()
    print("Selecting lactate masking on the original A/B Full folds", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [
            pool.submit(fit, schemas, "Full", str(fold), rate)
            for rate in CANDIDATES for fold in FOLD_IDS
        ]
        cv = [future.result()[1] for future in pending]
        candidates = {
            str(rate): aggregate([row for row in cv if row["recipe"] == str(rate)]) for rate in CANDIDATES
        }
        for row in candidates.values():
            row["ap_delta"] = {key: row["views"][key]["ap"] - reference_ap[key] for key in guard_views}
        gcs_ok = [k for k, row in candidates.items() if row["ap_delta"]["full_gcs"] >= -AP_TOLERANCE]
        both_ok = [k for k in gcs_ok if candidates[k]["ap_delta"]["clean"] >= -AP_TOLERANCE]
        stage = "both" if both_ok else "gcs_only" if gcs_ok else "all"
        selected = int(max(
            both_ok or gcs_ok or list(candidates),
            key=lambda key: (candidates[key]["mean_masked_ap"], -candidates[key]["mean_masked_lactate_rho"], -int(key)),
        ))
        print(f"Selected lactate masking={selected}% on A/B; eligibility={stage}", flush=True)
        elastic = [pool.submit(fit, schemas, "Elastic", str(fold), selected) for fold in FOLD_IDS]
        finals = {name: pool.submit(fit, schemas, name, "final", selected) for name in MODELS}
        elastic_rows = [future.result()[1] for future in elastic]
        cv.extend(elastic_rows)
        with tempfile.TemporaryDirectory(dir=OUT, prefix="ma-gbt-lactate-") as temporary:
            staging = Path(temporary) / OUTPUT.name
            staging.mkdir()
            fits, hashes = [], {}
            for name, future in finals.items():
                model, row = future.result()
                old = next(item for item in previous["final_fits"] if item["model"] == name)
                if {key: row["newly_masked"][key] for key in FIXED_RATES} != old["newly_masked"]:
                    raise ValueError("The fixed channel counts differ from P16")
                file = staging / f"{name}.joblib"
                joblib.dump(model, file)
                hashes[name] = sha(file)
                fits.append(row)
            verify_hashes(sources)
            result = {
                "training_dataset": "P12 A/B", "device": "cpu", "cpu_workers": workers,
                "alpha": ALPHA, "parameters": PARAMETERS, "upstream_commit": UPSTREAM_SHA,
                "fixed_rates_percent": FIXED_RATES, "lactate_candidate_rates_percent": list(CANDIDATES),
                "selected_lactate_rate_percent": selected, "mask_seed": MASK_SEED,
                "selection_eligibility": stage, "ap_guard_tolerance": AP_TOLERANCE, "reference_ap_full": reference_ap,
                "validation_joint_rates": VALIDATION_JOINT, "validation_joint_repeats": VALIDATION_REPEATS,
                "candidates_full": candidates, "elastic_cv_summary": aggregate(elastic_rows),
                "cv_folds": cv, "final_fits": fits, "source_sha256": sources, "model_sha256": hashes,
            }
            (staging / "manifest.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            staging.rename(OUTPUT)
    print("Frozen both lactate-augmented MA-GBT models fitted on A/B", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    train(parser.parse_args().workers)
