"""Tune and freeze the upstream MA-GBT classifier using the original P12 A/B folds."""

import argparse
import json
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from ma_gbt_p12_inputs import RUN, SCHEMA_FILE, encode

import joblib
import numpy as np
import torch
from malearn.estimators import regression
from malearn.estimators.classification import MAGBTClassifier
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, roc_auc_score

from prepare import OUT, sha
from train_missingness_augmentation import FOLD_IDS, MODELS, ab_inputs, from_json

UPSTREAM_SHA = "fbbdd0ca2c9d0d8072f49d82ad35f4391098863f"
PARAMETERS = {
    "n_estimators": 10, "loss": "log_loss", "learning_rate": 0.1,
    "max_depth": 2, "subsample": 1.0, "random_state": 42,
}
ALPHAS = (0.1, 10.0, 1000.0)
AUROC_FRACTION = 0.95
FITTED = RUN / "fitted"


class ScalarNumpy:
    """Delegate NumPy except for the author's tight finite-scalar split loop."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def isclose(a, b):
        # NumPy 1.25.2 promotes the second zero-dimensional argument to float64.
        # Encoded inputs are checked as finite float32 before entering author code.
        x, y = float(a), float(b)
        return abs(x - y) <= 1e-8 + 1e-5 * abs(y)


@contextmanager
def fast_scalar_splits():
    original = regression.np
    regression.np = ScalarNumpy()
    try:
        yield
    finally:
        regression.np = original


def validate_upstream():
    source = Path(__import__("malearn").__path__[0]).resolve().parent
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    modifications = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if (head != UPSTREAM_SHA or modifications or sklearn_version != "1.5.2"
            or np.__version__ != "1.25.2" or torch.version.cuda is not None):
        raise RuntimeError("Expected the pinned, unmodified author code and CPU-only runtime")


def check_scalar_splits(manifest):
    x, y, fold = ab_inputs()
    schema = manifest["schemas"]["Full"]["0"]
    xa, ma = encode(x.loc[fold != 0].iloc[:500], schema)
    xv, mv = encode(x.loc[fold == 0], schema)
    adjacent = np.sort(xa, axis=0)[::17].ravel()
    edges = np.nextafter(
        (adjacent.astype(float) + 1e-8 + 1e-5 * np.abs(adjacent.astype(float))).astype(np.float32),
        np.float32(-np.inf),
    )
    for right in (edges, np.nextafter(edges, np.float32(np.inf))):
        expected = [np.isclose(a, b) for a, b in zip(right, adjacent, strict=True)]
        actual = [ScalarNumpy.isclose(a, b) for a, b in zip(right, adjacent, strict=True)]
        np.testing.assert_array_equal(actual, expected)
    params = {**PARAMETERS, "n_estimators": 3, "alpha": 10.0}
    labels = y[fold != 0][:500]
    slow = MAGBTClassifier(**params).fit(xa, labels, M=ma)
    with fast_scalar_splits():
        fast = MAGBTClassifier(**params).fit(xa, labels, M=ma)
    for old, new in zip(slow.estimators_.ravel(), fast.estimators_.ravel(), strict=True):
        for field in ("feature", "threshold", "value", "children_left", "children_right"):
            np.testing.assert_array_equal(getattr(old.tree_, field), getattr(new.tree_, field))
    np.testing.assert_array_equal(slow.predict_proba(xv), fast.predict_proba(xv))
    np.testing.assert_equal(
        slow.compute_missingness_reliance(xv, mv), fast.compute_missingness_reliance(xv, mv)
    )
    print("Scalar comparisons, original and accelerated trees, probabilities and rho agree exactly", flush=True)


def train_one(manifest, name, fold, alpha):
    x, y, folds = ab_inputs()
    schema = manifest["schemas"][name][fold]
    fit_rows = np.ones(len(y), dtype=bool) if fold == "final" else folds != int(fold)
    features, mask = encode(x.loc[fit_rows], schema)
    begins = time.perf_counter()
    with fast_scalar_splits():
        model = MAGBTClassifier(alpha=alpha, **PARAMETERS).fit(features, y[fit_rows], M=mask)
    duration = time.perf_counter() - begins
    if list(model.classes_) != [0, 1]:
        raise RuntimeError("Unexpected class order in MA-GBT")
    metrics = {"model": name, "fold": fold, "alpha": alpha, "seconds": duration}
    if fold != "final":
        xv, mv = encode(x.loc[~fit_rows], schema)
        target = y[~fit_rows]
        predicted = model.predict_proba(xv)[:, 1]
        rho = float(model.compute_missingness_reliance(xv, mv))
        if not 0 <= rho <= 1:
            raise ValueError("Missingness reliance is outside [0, 1]")
        metrics.update(
            ap=float(average_precision_score(target, predicted)),
            auroc=float(roc_auc_score(target, predicted)),
            rho=rho,
        )
    print(json.dumps(metrics), flush=True)
    return model, metrics


def score_fold(manifest, name, fold, alpha):
    return train_one(manifest, name, fold, alpha)[1]


def train(workers):
    if FITTED.exists():
        raise FileExistsError(FITTED)
    schema_hash = sha(SCHEMA_FILE)
    manifest = from_json(SCHEMA_FILE)
    sources = manifest["source_sha256"]

    def verify_sources():
        if sha(SCHEMA_FILE) != schema_hash or any(sha(OUT / path) != digest for path, digest in sources.items()):
            raise ValueError("The frozen A/B inputs or preprocessing source have changed")

    verify_sources()
    check_scalar_splits(manifest)
    print(f"Running A/B Full three-fold CV with {workers} CPU workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(score_fold, manifest, "Full", str(fold), alpha)
            for alpha in ALPHAS for fold in FOLD_IDS
        ]
        rows = [future.result() for future in futures]
        candidates = []
        for alpha in ALPHAS:
            subset = [row for row in rows if row["alpha"] == alpha]
            candidates.append({
                "alpha": alpha,
                **{metric: float(np.mean([row[metric] for row in subset])) for metric in ("ap", "auroc", "rho")},
            })
        cutoff = AUROC_FRACTION * max(row["auroc"] for row in candidates)
        eligible = [row for row in candidates if row["auroc"] >= cutoff]
        winner = min(eligible, key=lambda row: (row["rho"], -row["auroc"], row["alpha"]))
        alpha = winner["alpha"]
        print(f"Selected alpha={alpha:g} using A/B Full only (AUROC floor={cutoff:.6f})", flush=True)
        elastic = [pool.submit(score_fold, manifest, "Elastic", str(fold), alpha) for fold in FOLD_IDS]
        final = {name: pool.submit(train_one, manifest, name, "final", alpha) for name in MODELS}
        rows.extend(future.result() for future in elastic)

        with tempfile.TemporaryDirectory(dir=RUN, prefix="ma-gbt-fit-") as temporary:
            staging = Path(temporary) / "fitted"
            staging.mkdir()
            final_metrics = []
            model_hashes = {}
            for name, future in final.items():
                model, metrics = future.result()
                path = staging / f"{name}.joblib"
                joblib.dump(model, path)
                model_hashes[name] = sha(path)
                final_metrics.append(metrics)
            verify_sources()
            result = {
                "model": "upstream MAGBTClassifier", "upstream_commit": UPSTREAM_SHA,
                "training_dataset": "P12 A/B", "device": "cpu", "cpu_workers": workers,
                "parameters": PARAMETERS, "cv_candidates_full": candidates,
                "numeric_check": "500 A/B training rows, 3 depth-2 trees, exact structure/predictions/rho",
                "auroc_floor_full": cutoff, "selected_alpha": alpha,
                "cv_folds": rows, "final_fits": final_metrics,
                "encoded_schemas_sha256": schema_hash, "model_sha256": model_hashes,
            }
            (staging / "manifest.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            staging.rename(FITTED)
    print("Frozen both A/B-fitted MA-GBT models", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--check", action="store_true", help="Verify scalar speedup on A/B")
    options = parser.parse_args()
    validate_upstream()
    if options.check:
        check_scalar_splits(from_json(SCHEMA_FILE))
    else:
        train(options.workers)


if __name__ == "__main__":
    main()
