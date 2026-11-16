"""Train GCS and urine missingness augmentation using P12 A/B only."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from prepare import OUT, sha
from stress_missingness_c import apply_masks, score_metrics


AB_FEATURES = OUT / "features_AB.parquet"
AB_LABELS = OUT / "labels_AB.csv"
AB_FOLDS = OUT / "folds3.csv"
AB_SELECTION = OUT / "selection_runs.json"
FROZEN = OUT / "final_models"
FOLD_IDS = (0, 1, 2)
MODELS = ("Full", "Elastic")
DEFAULT_OUTPUT = OUT / "missingness_augmentation_AB"
CANDIDATES = {
    "moderate": {"GCS": 25, "Urine": 75},
    "strong": {"GCS": 60, "Urine": 100},
}
VALIDATION_MASKS = {"GCS": 25, "Urine": 100}
FULL_MASKS = {"GCS": 100, "Urine": 100}
SEED = 2026111601
CLEAN_TOLERANCE = 0.01
CONDITIONS = ("clean", "partial_joint", "full_joint")


def from_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ab_inputs():
    x = pd.read_parquet(AB_FEATURES)
    y = pd.read_csv(AB_LABELS).set_index("RecordID")
    folds = pd.read_csv(AB_FOLDS).set_index("RecordID")["fold3"]
    if x.shape != (8000, 435) or not x.index.is_unique or not y.index.is_unique or not folds.index.is_unique:
        raise ValueError("Unexpected A/B shape or duplicated index")
    if not set(x.index) == set(y.index) == set(folds.index):
        raise ValueError("A/B features, outcomes and folds do not align")
    y = y.loc[x.index]
    folds = folds.loc[x.index]
    if set(y.source_set) != {"A", "B"} or set(folds) != set(FOLD_IDS):
        raise ValueError("Expected A/B sources and three original folds")
    values = y.In_hospital_death.to_numpy()
    if set(values) != {0, 1} or int(values.sum()) != 1122:
        raise ValueError("A/B outcome counts changed")
    return x, values, folds.to_numpy()


def verified_template(path, feature_sha):
    pipe, metadata, columns = joblib.load(path)
    model = pipe.named_steps["model"]
    if not isinstance(model, XGBClassifier) or pipe.classes_.tolist() != [0, 1]:
        raise TypeError("Unexpected original model or outcome class order")
    if metadata["feature_data_sha256"] != feature_sha or len(set(columns)) != len(columns):
        raise ValueError("Original model does not refer to the same A/B feature table")
    model.set_params(device="cpu", n_jobs=2)
    if model.get_xgb_params().get("device") != "cpu":
        raise RuntimeError("CPU selection failed")
    return pipe, columns


def mask_observed(features, rates, seed):
    rng = np.random.default_rng(seed)
    masks = {}
    plan = {}
    for variable, percent in rates.items():
        observed = np.flatnonzero(features[f"{variable}__count"].gt(0).to_numpy())
        count = (len(observed) * percent + 50) // 100
        masks[variable] = rng.permutation(observed)[:count]
        plan[variable] = {"set_c_matched_observed": len(observed) - count}
    masked = apply_masks(features, masks, plan, "matched")
    positions = np.unique(np.concatenate(list(masks.values())))
    return masked, positions, {variable: len(positions) for variable, positions in masks.items()}


def validation_views(features, fold):
    partial = [mask_observed(features, VALIDATION_MASKS, SEED + 100 * fold + repeat + 20)[0] for repeat in range(3)]
    full = mask_observed(features, FULL_MASKS, SEED + 100 * fold + 40)[0]
    return {"clean": [features], "partial_joint": partial, "full_joint": [full]}


def fit_augmented(template, columns, x, y, masked, changed):
    pre = template.named_steps["pre"]
    encoded = pre.transform(x.loc[:, columns])
    copies = pre.transform(masked.iloc[changed].loc[:, columns])
    weights = np.ones(len(x), dtype=float)
    weights[changed] = 0.5
    sample_weight = np.concatenate([weights, np.full(len(changed), 0.5)])
    labels = np.concatenate([y, y[changed]])
    if sample_weight.sum() != len(x) or sample_weight[labels == 1].sum() != y.sum():
        raise RuntimeError("Augmentation changed the total or positive training weight")
    model = clone(template.named_steps["model"]).set_params(device="cpu", n_jobs=2)
    start = time.perf_counter()
    model.fit(np.concatenate([encoded, copies]), labels, sample_weight=sample_weight)
    elapsed = time.perf_counter() - start
    return Pipeline([("pre", pre), ("model", model)]), elapsed


def evaluate_fold(pipe, columns, views, y):
    rows = []
    for condition, frames in views.items():
        metrics = [score_metrics(y, pipe.predict_proba(frame.loc[:, columns])[:, 1]) for frame in frames]
        rows.append({"condition": condition, **{key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}})
    return rows


def validate_oof(pipe, columns, x, y, fold, oof):
    original = oof.loc[x.index]
    if not original.fold.eq(fold).all() or not np.array_equal(original.y.to_numpy(), y):
        raise ValueError("Archived validation folds do not match the original fit split")
    score = pipe.predict_proba(x.loc[:, columns])[:, 1]
    np.testing.assert_array_equal(score, original.score.to_numpy(dtype=score.dtype))


def cv_family(name, candidates, x, y, folds, folder, feature_sha):
    cfg = from_json(folder / "config.json")
    expected = {"model": "XGBoost", "family": 2, "strategy": name, "nfold": 3, "adult": False, "data_sha": feature_sha}
    if any(cfg[key] != value for key, value in expected.items()):
        raise ValueError(f"Archived {name} run does not match the intended split and strategy")
    oof = pd.read_csv(folder / "oof.csv").set_index("RecordID")
    if not oof.index.is_unique or set(oof.index) != set(x.index):
        raise ValueError("Archived A/B prediction keys do not match")
    results = []
    fit_log = []
    for fold in FOLD_IDS:
        untouched = folds != fold
        train_x, train_y, valid_x, valid_y = x.loc[untouched], y[untouched], x.loc[~untouched], y[~untouched]
        template, columns = verified_template(folder / f"fold{fold}.joblib", feature_sha)
        validate_oof(template, columns, valid_x, valid_y, fold, oof)
        views = validation_views(valid_x, fold)
        results.extend({"model": name, "recipe": "frozen", "fold": fold, **r} for r in evaluate_fold(template, columns, views, valid_y))
        for recipe in candidates:
            masked, changed, hidden = mask_observed(train_x, CANDIDATES[recipe], SEED + 100 * fold)
            augmented, elapsed = fit_augmented(template, columns, train_x, train_y, masked, changed)
            results.extend({"model": name, "recipe": recipe, "fold": fold, **r} for r in evaluate_fold(augmented, columns, views, valid_y))
            fit_log.append({
                "model": name, "recipe": recipe, "fold": fold, "source_stays": len(train_x),
                "synthetic_copies": len(changed), "masked_counts": hidden, "fit_seconds": elapsed,
            })
            print(f"A/B {name} fold {fold} {recipe}: fitted with CPU in {elapsed:.1f}s", flush=True)
    return results, fit_log


def choose_recipe(results):
    rows = pd.DataFrame(results).groupby(["recipe", "condition"]).ap.mean().unstack()
    table = {}
    for name in ("frozen", *CANDIDATES):
        val = rows.loc[name]
        table[name] = {
            "clean_ap": float(val.clean),
            "partial_joint_ap": float(val.partial_joint),
            "full_joint_ap": float(val.full_joint),
            "masked_ap": float((val.partial_joint + val.full_joint) / 2),
            "clean_ap_delta": float(val.clean - rows.loc["frozen", "clean"]),
        }
    passed = [name for name in CANDIDATES if table[name]["clean_ap_delta"] >= -CLEAN_TOLERANCE]
    ranking = "masked_ap" if passed else "clean_ap"
    selected = max(passed or list(CANDIDATES), key=lambda name: table[name][ranking])
    return selected, bool(passed), table


def run(output):
    output = output.resolve()
    root = OUT.resolve()
    if output == root or not output.is_relative_to(root) or output.exists() or not output.parent.is_dir():
        raise ValueError("Use a new output directory under ML/artifacts")

    runs = from_json(AB_SELECTION)
    folders = {name: OUT / "runs" / runs[f"XGBoost_{name}"] for name in MODELS}
    originals = {name: FROZEN / f"{name}.joblib" for name in MODELS}
    sources = [AB_FEATURES, AB_LABELS, AB_FOLDS, AB_SELECTION, OUT / "fit_ledger.jsonl", FROZEN / "metadata.json", *originals.values()]
    sources += [
        folder / filename for folder in folders.values()
        for filename in ("config.json", "oof.csv", *(f"fold{fold}.joblib" for fold in FOLD_IDS))
    ]
    source_hashes = {path: sha(path) for path in sources}
    feature_sha = source_hashes[AB_FEATURES]
    x, y, folds = ab_inputs()

    with threadpool_limits(limits=2):
        full_rows, full_fits = cv_family("Full", CANDIDATES, x, y, folds, folders["Full"], feature_sha)
        selected, guard, full_cv = choose_recipe(full_rows)
        print(f"A/B Full selected {selected}. Clean guard passed: {guard}.", flush=True)
        elastic_rows, elastic_fits = cv_family("Elastic", (selected,), x, y, folds, folders["Elastic"], feature_sha)

        fit_log = full_fits + elastic_fits
        masked, changed, hidden = mask_observed(x, CANDIDATES[selected], SEED + 1000)
        final_meta = from_json(FROZEN / "metadata.json")
        with tempfile.TemporaryDirectory(dir=output.parent, prefix="ab-augmentation-") as temporary:
            directory = Path(temporary).resolve() / "run"
            if directory.parent.parent != output.parent:
                raise RuntimeError("Unexpected temporary output path")
            directory.mkdir()
            model_hashes = {}
            for name, path in originals.items():
                template, columns = verified_template(path, feature_sha)
                if columns != final_meta[name]["columns"]:
                    raise ValueError("The all-A/B feature selection differs from the archived metadata")
                pipe, elapsed = fit_augmented(template, columns, x, y, masked, changed)
                info = {
                    "model": name, "recipe": selected, "fold": "all_AB", "source_stays": len(x),
                    "synthetic_copies": len(changed), "masked_counts": hidden, "fit_seconds": elapsed,
                }
                fit_log.append(info)
                path = directory / f"{name}.joblib"
                joblib.dump((pipe, info, columns), path)
                model_hashes[name] = sha(path)
                print(f"Frozen augmented {name}: fitted on all A/B in {elapsed:.1f}s", flush=True)

            if len(fit_log) != 11 or any(sha(path) != digest for path, digest in source_hashes.items()):
                raise RuntimeError("Unexpected fit count or a changed A/B input")
            manifest = {
                "training_cohort": "P12 A/B", "stays": len(x), "deaths": int(y.sum()),
                "device": "cpu", "threads": 2, "seed": SEED, "selected_recipe": selected,
                "selected_rates_percent": CANDIDATES[selected], "full_guard_passed": guard,
                "full_cv": full_cv, "new_fits": len(fit_log),
                "original_model_sha256": {name: source_hashes[path] for name, path in originals.items()},
                "model_sha256": model_hashes,
                "source_sha256": {str(path.relative_to(OUT)).replace(os.sep, "/"): value for path, value in source_hashes.items()},
            }
            pd.DataFrame(full_rows + elastic_rows).to_csv(directory / "cv_metrics.csv", index=False)
            (directory / "fit_log.json").write_text(json.dumps(fit_log, indent=2) + "\n", encoding="utf-8")
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            directory.rename(output)
    print(f"Saved frozen A/B augmentation models and manifest under {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
