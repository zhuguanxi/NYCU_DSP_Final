"""Fit fixed alpha=10 MA-GBT with and without the P14 augmentation."""

import argparse
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE

import joblib
import numpy as np

from evaluate_ma_gbt_p12 import FITTED as P13_FITTED, check_fit
from prepare import OUT, sha
from run_ma_gbt_p12 import PARAMETERS, UPSTREAM_SHA, validate_upstream
from train_ma_gbt_augmentation import MASK_SEED, OUTPUT as P14_FITTED, RECIPES, fit_one, summarize
from train_missingness_augmentation import FOLD_IDS, MODELS, from_json

OUTPUT = OUT / "ma_gbt_alpha10"
ALPHA = 10.0
PREVIOUS_ALPHA = 1000.0
RATES = {"GCS": 45, "ICUType": 75}
RECIPES_BY_RUN = {"ma10": None, "aug10": "medium"}


def verify_hashes(digests):
    for relative, value in digests.items():
        if sha(OUT / relative) != value:
            raise ValueError(f"Source artifact changed: {relative}")


def inputs():
    original = check_fit()
    p14_manifest = P14_FITTED / "manifest.json"
    augmented = from_json(p14_manifest)
    schema = from_json(SCHEMA_FILE)
    if original["selected_alpha"] != PREVIOUS_ALPHA or augmented["alpha"] != PREVIOUS_ALPHA:
        raise ValueError("Expected alpha=1000 in both historical comparison groups")
    if any(fit["parameters"] != PARAMETERS or fit["upstream_commit"] != UPSTREAM_SHA for fit in (original, augmented)):
        raise ValueError("Historical runs used different MA-GBT settings")
    if (
        RECIPES["medium"] != RATES or augmented["selected_rates_percent"] != RATES
        or augmented["mask_seed"] != MASK_SEED or augmented["selected_recipe"] != "medium"
        or augmented["original_model_sha256"] != original["model_sha256"]
    ):
        raise ValueError("Expected the frozen 45% GCS and 75% ICU type augmentation")
    digests = {
        **schema["source_sha256"],
        SCHEMA_FILE.relative_to(OUT).as_posix(): original["encoded_schemas_sha256"],
        (P13_FITTED / "manifest.json").relative_to(OUT).as_posix(): augmented["original_manifest_sha256"],
        p14_manifest.relative_to(OUT).as_posix(): sha(p14_manifest),
        **{(P14_FITTED / f"{name}.joblib").relative_to(OUT).as_posix(): augmented["model_sha256"][name] for name in MODELS},
    }
    verify_hashes(digests)
    return schema["schemas"], original, augmented, digests


def check_ab_reference(cv, original):
    saved = {
        row["fold"]: row for row in original["cv_folds"]
        if row["model"] == "Full" and row["alpha"] == ALPHA
    }
    plain_full = [row for row in cv["ma10"] if row["model"] == "Full"]
    if len(plain_full) != len(saved) or set(saved) != set(map(str, FOLD_IDS)):
        raise ValueError("Expected the original three P13 alpha=10 Full validation folds")
    for row in plain_full:
        for metric in ("ap", "auroc", "rho"):
            np.testing.assert_allclose(row["views"]["clean"][metric], saved[row["fold"]][metric], rtol=0, atol=1e-12)
    print("Reproduced the three original A/B Full alpha=10 validation folds", flush=True)


def train(workers):
    validate_upstream()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    schemas, original, augmented, digests = inputs()
    print("Running fixed alpha=10 A/B CV for both variants and feature sets", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {
            tag: [pool.submit(fit_one, schemas, name, str(fold), recipe, ALPHA)
                  for name in MODELS for fold in FOLD_IDS]
            for tag, recipe in RECIPES_BY_RUN.items()
        }
        cv = {tag: [future.result()[1] for future in futures] for tag, futures in pending.items()}
        check_ab_reference(cv, original)

        finals = {
            (tag, name): pool.submit(fit_one, schemas, name, "final", recipe, ALPHA)
            for tag, recipe in RECIPES_BY_RUN.items() for name in MODELS
        }
        with tempfile.TemporaryDirectory(dir=OUT, prefix="ma-gbt-alpha10-") as temporary:
            staging = Path(temporary) / OUTPUT.name
            staging.mkdir()
            for tag in RECIPES_BY_RUN:
                (staging / tag).mkdir()
            fits, model_hashes = {}, {}
            for (tag, name), future in finals.items():
                model, stats = future.result()
                if tag == "aug10":
                    old = next(row for row in augmented["final_fits"] if row["model"] == name)
                    if stats["newly_masked"] != old["newly_masked"]:
                        raise ValueError("Final augmentation counts differ from P14")
                file = staging / tag / f"{name}.joblib"
                joblib.dump(model, file)
                model_hashes[file.relative_to(staging).as_posix()] = sha(file)
                fits.setdefault(tag, []).append(stats)
            verify_hashes(digests)
            check_fit()
            summary = {
                tag: {name: summarize([row for row in rows if row["model"] == name]) for name in MODELS}
                for tag, rows in cv.items()
            }
            manifest = {
                "alpha": ALPHA, "parameters": PARAMETERS, "upstream_commit": UPSTREAM_SHA,
                "augmentation_rates_percent": RATES, "mask_seed": MASK_SEED,
                "training_dataset": "P12 A/B", "tuned_in_this_run": False,
                "device": "cpu", "cpu_workers": workers, "folds": list(FOLD_IDS),
                "cv_folds": cv, "cv_summary": summary, "final_fits": fits,
                "source_sha256": digests, "model_sha256": model_hashes,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            staging.rename(OUTPUT)
    print("Frozen all four alpha=10 models fitted on A/B", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    train(parser.parse_args().workers)
