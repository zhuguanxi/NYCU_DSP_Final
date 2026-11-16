"""Select GCS and ICU type masking on P12 A/B and freeze augmented MA-GBT."""

import argparse
import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE, encode

import joblib
import numpy as np
from malearn.estimators.classification import MAGBTClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_ma_gbt_p12 import FITTED as ORIGINAL_MODELS, check_fit
from prepare import OUT, sha
from run_ma_gbt_p12 import PARAMETERS, UPSTREAM_SHA, fast_scalar_splits, validate_upstream
from stress_missingness_c import apply_masks
from train_missingness_augmentation import FOLD_IDS, MODELS, ab_inputs, from_json

OUTPUT = OUT / "ma_gbt_gcs_icu_augmentation"
RECIPES = {
    "light": {"GCS": 25, "ICUType": 50},
    "medium": {"GCS": 45, "ICUType": 75},
    "strong": {"GCS": 60, "ICUType": 90},
}
VALIDATION_PARTIAL = {"GCS": 25, "ICUType": 90}
VALIDATION_REPEATS = 3
CLEAN_AP_TOLERANCE = 0.01
MASK_SEED = 2026111603
ICU_TYPE = "static_ICUType"


def mask_features(frame, rates, seed):
    rng = np.random.default_rng(seed)
    observed = {
        "GCS": np.flatnonzero(frame["GCS__count"].gt(0).to_numpy()),
        "ICUType": np.flatnonzero(frame[ICU_TYPE].notna().to_numpy()),
    }
    if "Urine" in rates:
        observed["Urine"] = np.flatnonzero(frame["Urine__count"].gt(0).to_numpy())
    selected = {
        key: rng.permutation(rows)[: (len(rows) * rates[key] + 50) // 100]
        for key, rows in observed.items()
    }
    dynamic = {key: rows for key, rows in selected.items() if key != "ICUType"}
    augmented = apply_masks(
        frame,
        dynamic,
        {key: {"set_c_matched_observed": len(observed[key]) - len(rows)} for key, rows in dynamic.items()},
        "matched",
    )
    augmented.loc[augmented.index[selected["ICUType"]], ICU_TYPE] = np.nan
    if len(observed["ICUType"]) - augmented[ICU_TYPE].count() != len(selected["ICUType"]):
        raise ValueError("ICU type masking did not remove exactly the selected observed values")
    if not augmented.index.equals(frame.index):
        raise ValueError("Augmentation changed the stay order or cohort")
    return augmented, {key: len(rows) for key, rows in selected.items()}


def validation_scores(model, frame, target, schema, fold):
    def score(view):
        features, mask = encode(view, schema)
        predictions = model.predict_proba(features)[:, 1]
        return [
            float(average_precision_score(target, predictions)),
            float(roc_auc_score(target, predictions)),
            float(model.compute_missingness_reliance(features, mask)),
        ]

    rows = {"clean": score(frame)}
    partial = [
        score(mask_features(frame, VALIDATION_PARTIAL, np.random.SeedSequence([MASK_SEED, fold, 1, i]))[0])
        for i in range(VALIDATION_REPEATS)
    ]
    rows["partial_joint"] = np.mean(partial, axis=0).tolist()
    for name, rates in {
        "full_gcs": {"GCS": 100, "ICUType": 0},
        "full_icutype": {"GCS": 0, "ICUType": 100},
        "full_joint": {"GCS": 100, "ICUType": 100},
    }.items():
        rows[name] = score(mask_features(frame, rates, np.random.SeedSequence([MASK_SEED, fold, 2]))[0])
    return {name: dict(zip(("ap", "auroc", "rho"), metrics, strict=True)) for name, metrics in rows.items()}


def fit_one(schemas, name, fold, recipe, alpha, *, rates=None, validation_fn=validation_scores):
    frame, target, folds = ab_inputs()
    schema = schemas[name][fold]
    is_final = fold == "final"
    fold_no = len(FOLD_IDS) if is_final else int(fold)
    train_rows = np.ones(len(target), dtype=bool) if is_final else folds != fold_no
    train = frame.loc[train_rows]
    removed = {"GCS": 0, "ICUType": 0}
    if rates is None and recipe is not None:
        rates = RECIPES[recipe]
    if rates is not None:
        train, removed = mask_features(
            train, rates, np.random.SeedSequence([MASK_SEED, fold_no, 0])
        )
    x, missing = encode(train, schema)
    elapsed = time.perf_counter()
    with fast_scalar_splits():
        estimator = MAGBTClassifier(alpha=alpha, **PARAMETERS).fit(x, target[train_rows], M=missing)
    elapsed = time.perf_counter() - elapsed
    if list(estimator.classes_) != [0, 1]:
        raise ValueError("Unexpected MA-GBT outcome ordering")
    row = {
        "model": name, "fold": fold, "recipe": recipe, "stays": len(train),
        "deaths": int(target[train_rows].sum()), "newly_masked": removed, "seconds": elapsed,
    }
    if not is_final:
        row["views"] = validation_fn(estimator, frame.loc[~train_rows], target[~train_rows], schema, fold_no)
    print(json.dumps(row), flush=True)
    return estimator if is_final else None, row


def summarize(rows):
    metrics = {
        view: {metric: float(np.mean([row["views"][view][metric] for row in rows])) for metric in ("ap", "auroc", "rho")}
        for view in rows[0]["views"]
    }
    stress_ap = float(np.mean([values["ap"] for view, values in metrics.items() if view != "clean"]))
    return {"views": metrics, "mean_masked_ap": stress_ap}


def train(workers):
    validate_upstream()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = check_fit()
    original_path = ORIGINAL_MODELS / "manifest.json"
    schema_digest, original_digest = sha(SCHEMA_FILE), sha(original_path)
    manifest = from_json(SCHEMA_FILE)
    source_hashes = manifest["source_sha256"]
    schemas = manifest["schemas"]
    alpha = original["selected_alpha"]
    if original["parameters"] != PARAMETERS or original["upstream_commit"] != UPSTREAM_SHA:
        raise ValueError("The frozen MA-GBT uses a different estimator or configuration")

    def verify():
        if sha(SCHEMA_FILE) != schema_digest or sha(original_path) != original_digest:
            raise ValueError("The original MA-GBT manifest or schema changed")
        if any(sha(OUT / file) != digest for file, digest in source_hashes.items()):
            raise ValueError("A frozen A/B input changed")
        check_fit()

    clean_rows = [
        row["ap"] for row in original["cv_folds"] if row["model"] == "Full" and row["alpha"] == alpha
    ]
    if len(clean_rows) != len(FOLD_IDS):
        raise ValueError("Missing frozen Full A/B validation metrics")
    original_clean_ap = float(np.mean(clean_rows))
    verify()
    print(f"Running {len(RECIPES)} augmentation strengths on original A/B Full folds", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fit_one, schemas, "Full", str(fold), recipe, alpha)
            for recipe in RECIPES for fold in FOLD_IDS
        ]
        cv = [future.result()[1] for future in futures]
        candidates = {
            recipe: summarize([row for row in cv if row["recipe"] == recipe]) for recipe in RECIPES
        }
        for candidate in candidates.values():
            candidate["clean_ap_delta"] = candidate["views"]["clean"]["ap"] - original_clean_ap
            candidate["passes_guard"] = candidate["clean_ap_delta"] >= -CLEAN_AP_TOLERANCE
        eligible = [recipe for recipe, row in candidates.items() if row["passes_guard"]]
        selected = max(
            eligible or list(RECIPES),
            key=lambda recipe: (candidates[recipe]["mean_masked_ap"], candidates[recipe]["views"]["clean"]["ap"]),
        )
        print(f"Selected {selected} using A/B only, clean AP guard passed={bool(eligible)}", flush=True)
        elastic = [pool.submit(fit_one, schemas, "Elastic", str(fold), selected, alpha) for fold in FOLD_IDS]
        finals = {name: pool.submit(fit_one, schemas, name, "final", selected, alpha) for name in MODELS}
        elastic_cv = [future.result()[1] for future in elastic]
        cv.extend(elastic_cv)

        with tempfile.TemporaryDirectory(dir=OUT, prefix="ma-gbt-aug-") as temporary:
            stages = Path(temporary) / "fitted"
            stages.mkdir()
            final_rows, model_hashes = [], {}
            for name, future in finals.items():
                model, row = future.result()
                path = stages / f"{name}.joblib"
                joblib.dump(model, path)
                final_rows.append(row)
                model_hashes[name] = sha(path)
            verify()
            result = {
                "training_dataset": "P12 A/B", "device": "cpu", "cpu_workers": workers,
                "upstream_commit": UPSTREAM_SHA, "parameters": PARAMETERS, "alpha": alpha,
                "augmentation": "one view per original training stay",
                "candidate_rates_percent": RECIPES, "mask_seed": MASK_SEED,
                "validation_partial_rates_percent": VALIDATION_PARTIAL, "validation_repeats": VALIDATION_REPEATS,
                "original_clean_ap_full": original_clean_ap, "clean_ap_tolerance": CLEAN_AP_TOLERANCE,
                "selected_recipe": selected, "selected_rates_percent": RECIPES[selected],
                "clean_ap_guard_passed": bool(eligible), "full_candidates": candidates,
                "elastic_cv": summarize(elastic_cv), "cv_folds": cv, "final_fits": final_rows,
                "encoded_schemas_sha256": schema_digest, "original_manifest_sha256": original_digest,
                "original_model_sha256": original["model_sha256"], "model_sha256": model_hashes,
            }
            (stages / "manifest.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            stages.rename(OUTPUT)
    print("Frozen A/B-only augmented MA-GBT Full and Elastic", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    train(parser.parse_args().workers)
