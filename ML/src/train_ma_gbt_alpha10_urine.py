"""Select urine masking on A/B with fixed alpha, GCS and ICU type masking."""

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
from train_ma_gbt_alpha10 import ALPHA, OUTPUT as P15_FITTED, RATES, verify_hashes
from train_ma_gbt_augmentation import MASK_SEED, VALIDATION_PARTIAL, VALIDATION_REPEATS, fit_one, mask_features
from train_missingness_augmentation import FOLD_IDS, MODELS, ab_inputs, from_json

OUTPUT = OUT / "ma_gbt_alpha10_urine"
URINE_CANDIDATES = (25, 75, 100)
CLEAN_AP_TOLERANCE = 0.01
VALIDATION_JOINT = {**VALIDATION_PARTIAL, "Urine": 100}
METRICS = ("ap", "auroc", "urine_rho")


def urine_mask(mask, schema):
    indices = [i for i, column in enumerate(schema["numeric"]) if column.startswith("Urine__")]
    filtered = np.zeros_like(mask)
    filtered[:, indices] = mask[:, indices]
    return filtered


def validation_scores(model, frame, target, schema, fold):
    def score(view):
        features, mask = encode(view, schema)
        predictions = model.predict_proba(features)[:, 1]
        return [
            float(average_precision_score(target, predictions)),
            float(roc_auc_score(target, predictions)),
            float(model.compute_missingness_reliance(features, urine_mask(mask, schema))),
        ]

    rows = {"clean": score(frame)}
    for name, rates in (
        ("full_urine", {"GCS": 0, "ICUType": 0, "Urine": 100}),
        ("full_gcs", {"GCS": 100, "ICUType": 0, "Urine": 0}),
    ):
        masked, _ = mask_features(frame, rates, np.random.SeedSequence([MASK_SEED, fold, 2]))
        rows[name] = score(masked)
    joint = [
        score(mask_features(frame, VALIDATION_JOINT, np.random.SeedSequence([MASK_SEED, fold, 1, i]))[0])
        for i in range(VALIDATION_REPEATS)
    ]
    rows["joint"] = np.mean(joint, axis=0).tolist()
    return {name: dict(zip(METRICS, values, strict=True)) for name, values in rows.items()}


def sources_for_p15():
    path = P15_FITTED / "manifest.json"
    saved = from_json(path)
    if (
        saved["alpha"] != ALPHA or saved["parameters"] != PARAMETERS
        or saved["upstream_commit"] != UPSTREAM_SHA or saved["augmentation_rates_percent"] != RATES
        or saved["mask_seed"] != MASK_SEED or (saved["training_dataset"], saved["device"]) != ("P12 A/B", "cpu")
        or saved["folds"] != list(FOLD_IDS)
    ):
        raise ValueError("P15 did not use the expected frozen A/B setup")
    source = {
        **saved["source_sha256"],
        path.relative_to(OUT).as_posix(): sha(path),
        **{(P15_FITTED / name).relative_to(OUT).as_posix(): digest for name, digest in saved["model_sha256"].items()},
    }
    verify_hashes(source)
    return saved, source


def check_masks():
    frame, _, _ = ab_inputs()
    probe = frame.iloc[:200]
    seed = np.random.SeedSequence([MASK_SEED, 0, 0])
    original, counts = mask_features(probe, RATES, seed)
    masked, extra = mask_features(probe, {**RATES, "Urine": 100}, seed)
    other = [column for column in probe if not column.startswith("Urine__")]
    pd.testing.assert_frame_equal(original[other], masked[other], check_exact=True)
    if {channel: extra[channel] for channel in RATES} != counts:
        raise ValueError("Adding urine masking changed the fixed channel counts")
    if not (masked["Urine__count"].eq(0).all() and masked["Urine__missing"].eq(1).all()):
        raise ValueError("The urine mask did not remove all observed values")
    print("A/B masking probe preserves all non-urine fields exactly", flush=True)


def aggregate(rows):
    means = {
        view: {metric: float(np.mean([row["views"][view][metric] for row in rows])) for metric in METRICS}
        for view in rows[0]["views"]
    }
    return {
        "views": means,
        "mean_masked_ap": (means["full_urine"]["ap"] + means["joint"]["ap"]) / 2,
        "mean_masked_urine_rho": (means["full_urine"]["urine_rho"] + means["joint"]["urine_rho"]) / 2,
    }


def fit(schemas, name, fold, strength):
    return fit_one(
        schemas, name, fold, str(strength), ALPHA,
        rates={**RATES, "Urine": strength}, validation_fn=validation_scores,
    )


def train(workers):
    validate_upstream()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior, sources = sources_for_p15()
    schemas = from_json(SCHEMA_FILE)["schemas"]
    reference_ap = prior["cv_summary"]["aug10"]["Full"]["views"]["clean"]["ap"]
    check_masks()
    print("Selecting urine masking on the three original Full A/B folds", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {
            strength: [pool.submit(fit, schemas, "Full", str(fold), strength) for fold in FOLD_IDS]
            for strength in URINE_CANDIDATES
        }
        cv = [future.result()[1] for futures in pending.values() for future in futures]
        candidates = {
            str(strength): aggregate([row for row in cv if row["recipe"] == str(strength)])
            for strength in URINE_CANDIDATES
        }
        for row in candidates.values():
            row["clean_ap_delta"] = row["views"]["clean"]["ap"] - reference_ap
            row["passes_guard"] = row["clean_ap_delta"] >= -CLEAN_AP_TOLERANCE
        eligible = [key for key, row in candidates.items() if row["passes_guard"]]
        selected = int(max(
            eligible or list(candidates),
            key=lambda key: (candidates[key]["mean_masked_ap"], -candidates[key]["mean_masked_urine_rho"], -int(key)),
        ))
        print(f"Selected urine masking={selected}% on A/B only, clean AP guard passed={bool(eligible)}", flush=True)
        elastic = [pool.submit(fit, schemas, "Elastic", str(fold), selected) for fold in FOLD_IDS]
        finals = {name: pool.submit(fit, schemas, name, "final", selected) for name in MODELS}
        elastic_results = [future.result()[1] for future in elastic]
        cv.extend(elastic_results)
        with tempfile.TemporaryDirectory(dir=OUT, prefix="ma-gbt-urine-") as temporary:
            staging = Path(temporary) / OUTPUT.name
            staging.mkdir()
            fits, hashes = [], {}
            for name, future in finals.items():
                model, row = future.result()
                saved = next(old for old in prior["final_fits"]["aug10"] if old["model"] == name)
                if {key: row["newly_masked"][key] for key in RATES} != saved["newly_masked"]:
                    raise ValueError("The fixed GCS or ICU type mask count differs from P15")
                file = staging / f"{name}.joblib"
                joblib.dump(model, file)
                hashes[name] = sha(file)
                fits.append(row)
            verify_hashes(sources)
            result = {
                "training_dataset": "P12 A/B", "device": "cpu", "cpu_workers": workers,
                "alpha": ALPHA, "parameters": PARAMETERS, "upstream_commit": UPSTREAM_SHA,
                "fixed_rates_percent": RATES, "urine_candidate_rates_percent": list(URINE_CANDIDATES),
                "selected_urine_rate_percent": selected, "mask_seed": MASK_SEED, "clean_ap_guard_passed": bool(eligible),
                "clean_ap_tolerance": CLEAN_AP_TOLERANCE, "reference_clean_ap_full": reference_ap,
                "validation_joint_rates": VALIDATION_JOINT, "validation_joint_repeats": VALIDATION_REPEATS,
                "candidates_full": candidates, "elastic_cv_summary": aggregate(elastic_results),
                "cv_folds": cv, "final_fits": fits, "source_sha256": sources, "model_sha256": hashes,
            }
            (staging / "manifest.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            staging.rename(OUTPUT)
    print("Frozen both augmented MA-GBT models fitted on A/B", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    train(parser.parse_args().workers)
