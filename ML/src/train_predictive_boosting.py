"""Select CPU boosting candidates on P12 A/B with fixed missingness conditions."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"

import joblib
import lightgbm
import numpy as np
import pandas as pd
import xgboost

from prepare import OUT, sha
from train_missingness_augmentation import (
    AB_FEATURES, AB_FOLDS, AB_LABELS, AB_SELECTION, FOLD_IDS, FROZEN, GCS_ONLY_OUTPUT,
    SEED as GCS_SEED, ab_inputs, fit_augmented, from_json, mask_observed, validate_oof, verified_template,
)
from stress_missingness_c import score_metrics

OUTPUT = OUT / "p21_predictive_boosting"
ANCHORS = ("frozen_full", "gcs_only_full")
UNMAPPED = ("Mg", "Urine", "MechVent")
SHIFT = {**dict.fromkeys(UNMAPPED, 100), "GCS": 25, "Lactate": 45, "MAP": 70}
STATIC_SHIFT = 90
CATEGORIES = {"static_Gender": (0, 1), "static_ICUType": (1, 2, 3, 4)}
VIEWS = ("clean", "shift1", "shift2", "full_gcs")
SHIFT_SEED = 2026111701
THREADS = 4
ROUNDS = (300, 650)
MODEL_PARAMS = {
    "xgb3": {"max_depth": 3, "min_child_weight": 1, "reg_lambda": 3},
    "xgb5": {"max_depth": 5, "min_child_weight": 5, "reg_lambda": 6},
    "lgb15": {"num_leaves": 15, "min_child_samples": 35, "reg_lambda": 2},
    "lgb31": {"num_leaves": 31, "min_child_samples": 55, "reg_lambda": 4},
    "cat5": {"depth": 5, "l2_leaf_reg": 4},
    "cat7": {"depth": 7, "l2_leaf_reg": 8},
}
POLICIES = (
    ("full", False), ("aligned", False), ("aligned", True),
)
FULL_AUGMENT = ("xgb3", "cat5")


def native_inputs(frame, scope):
    omit = set(UNMAPPED) if scope == "aligned" else set()
    columns = [key for key in frame if key.split("__", 1)[0] not in omit and key not in CATEGORIES]
    values = frame.loc[:, columns].to_numpy(dtype=np.float32)
    pieces = [values]
    for key, codes in CATEGORIES.items():
        raw = frame[key].to_numpy(dtype=np.float32)
        pieces.append(np.where(np.isnan(raw[:, None]), np.nan, raw[:, None] == codes).astype(np.float32))
    result = np.ascontiguousarray(np.column_stack(pieces), dtype=np.float32)
    if np.isinf(result).any():
        raise ValueError("The frozen features contain an infinite value")
    return result


def shift(frame, seed):
    masked, _, _ = mask_observed(frame, SHIFT, seed)
    original = np.flatnonzero(frame["static_ICUType"].notna().to_numpy())
    rng = np.random.default_rng(seed + 1)
    n = (len(original) * STATIC_SHIFT + 50) // 100
    masked.loc[masked.index[rng.permutation(original)[:n]], "static_ICUType"] = np.nan
    return masked


def candidate_specs(algorithms):
    return [
        {"algorithm": alg, "scope": scope, "augment": aug}
        for alg in algorithms
        for scope, aug in (*POLICIES, *(("full", True),) * (alg in FULL_AUGMENT))
    ]


def estimator(algorithm, rounds):
    common = MODEL_PARAMS[algorithm]
    if algorithm.startswith("xgb"):
        return xgboost.XGBClassifier(
            n_estimators=rounds, learning_rate=0.035, subsample=0.9, colsample_bytree=0.8,
            tree_method="hist", device="cpu", n_jobs=THREADS, random_state=42, **common,
        )
    if algorithm.startswith("lgb"):
        return lightgbm.LGBMClassifier(
            n_estimators=rounds, learning_rate=0.035, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, verbosity=-1, deterministic=True, force_col_wise=True,
            device_type="cpu", n_jobs=THREADS, random_state=42, **common,
        )
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=rounds, learning_rate=0.04, random_seed=42, task_type="CPU",
        thread_count=THREADS, verbose=False, allow_writing_files=False, **common,
    )


def native_fit(spec, matrix, y, shifted, rounds):
    model = estimator(spec["algorithm"], rounds)
    if spec["augment"]:
        count = len(y)
        matrix = np.concatenate((matrix, shifted))
        weights = np.repeat((0.75, 0.25), count)
        y = np.tile(y, 2)
    else:
        weights = None
    model.fit(matrix, y, sample_weight=weights)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected outcome class ordering")
    return model


def native_predict(model, spec, matrix, rounds):
    alg = spec["algorithm"]
    if alg.startswith("xgb"):
        score = model.predict_proba(matrix, iteration_range=(0, rounds))[:, 1]
    elif alg.startswith("lgb"):
        score = model.booster_.predict(matrix, num_iteration=rounds, num_threads=THREADS)
    else:
        score = model.predict_proba(matrix, ntree_end=rounds, thread_count=THREADS)[:, 1]
    return np.asarray(score, dtype=float)


def candidate_id(spec, rounds):
    return f"{spec['algorithm']}_{spec['scope']}_{'aug' if spec['augment'] else 'plain'}_{rounds}"


def source_paths():
    base = OUT / "runs" / from_json(AB_SELECTION)["XGBoost_Full"]
    return base, [
        AB_FEATURES, AB_LABELS, AB_FOLDS, AB_SELECTION, base / "oof.csv", base / "config.json",
        *(base / f"fold{k}.joblib" for k in FOLD_IDS),
        FROZEN / "Full.joblib", GCS_ONLY_OUTPUT / "Full.joblib", GCS_ONLY_OUTPUT / "manifest.json",
    ]


def average_metrics(y, folds, scores):
    ap = {}
    for view, predicted in scores.items():
        ap[view] = float(np.mean([
            score_metrics(y[folds == fold], predicted[folds == fold])["ap"] for fold in FOLD_IDS
        ]))
    ap["shift"] = (ap.pop("shift1") + ap.pop("shift2")) / 2
    ap["balanced"] = 0.65 * ap["clean"] + 0.25 * ap["shift"] + 0.10 * ap["full_gcs"]
    return ap


def verify_sources(sources):
    for path, digest in sources.items():
        if sha(OUT / path) != digest:
            raise ValueError(f"A/B source changed: {path}")


def train(algorithms):
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    x, y, folds = ab_inputs()
    xgb_run, inputs = source_paths()
    sources = {path.relative_to(OUT).as_posix(): sha(path) for path in inputs}
    feature_sha = sha(AB_FEATURES)
    gcs_manifest = from_json(GCS_ONLY_OUTPUT / "manifest.json")
    if gcs_manifest["selected_rates_percent"] != {"GCS": 60}:
        raise ValueError("Expected the frozen P11 GCS-only Full recipe")
    if (
        gcs_manifest["original_model_sha256"]["Full"] != sha(FROZEN / "Full.joblib")
        or gcs_manifest["model_sha256"]["Full"] != sha(GCS_ONLY_OUTPUT / "Full.joblib")
    ):
        raise ValueError("Frozen full-model artifacts differ from P11")
    archived = pd.read_csv(xgb_run / "oof.csv").set_index("RecordID")
    specs = candidate_specs(algorithms)
    definitions = {candidate_id(spec, n): {**spec, "rounds": n} for spec in specs for n in ROUNDS}
    names = (*ANCHORS, *definitions)
    observed = {key: {view: np.full(len(y), np.nan) for view in VIEWS} for key in names}
    fit_times = []
    for fold in FOLD_IDS:
        trained = folds != fold
        train_x, held_x = x.loc[trained], x.loc[~trained]
        train_y, held_y = y[trained], y[~trained]
        validation = {
            "clean": held_x, "shift1": shift(held_x, SHIFT_SEED + 100 * fold + 20),
            "shift2": shift(held_x, SHIFT_SEED + 100 * fold + 30),
            "full_gcs": mask_observed(held_x, {"GCS": 100}, SHIFT_SEED + fold)[0],
        }
        template, cols = verified_template(xgb_run / f"fold{fold}.joblib", feature_sha)
        validate_oof(template, cols, held_x, held_y, fold, archived)
        masked, changed, _ = mask_observed(train_x, {"GCS": 60}, GCS_SEED + 100 * fold)
        gcs, seconds = fit_augmented(template, cols, train_x, train_y, masked, changed)
        fit_times.append({"fold": fold, "name": "gcs_only_full", "seconds": seconds})
        for tag, model in zip(ANCHORS, (template, gcs), strict=True):
            for key, frame in validation.items():
                observed[tag][key][~trained] = model.predict_proba(frame.loc[:, cols])[:, 1]
        train_shifted = shift(train_x, SHIFT_SEED + 100 * fold)
        cases = {
            scope: (native_inputs(train_x, scope), native_inputs(train_shifted, scope),
                    {key: native_inputs(frame, scope) for key, frame in validation.items()})
            for scope in ("full", "aligned")
        }
        for spec in specs:
            t = time.perf_counter()
            mat, synthetic, eval_views = cases[spec["scope"]]
            model = native_fit(spec, mat, train_y, synthetic, ROUNDS[-1])
            for rounds in ROUNDS:
                tag = candidate_id(spec, rounds)
                for key, values in eval_views.items():
                    observed[tag][key][~trained] = native_predict(model, spec, values, rounds)
            elapsed = time.perf_counter() - t
            fit_times.append({"fold": fold, "name": tag, "seconds": elapsed})
            print(f"A/B fold {fold} {tag}, {elapsed:.1f}s", flush=True)

    scores = {tag: average_metrics(y, folds, pred) for tag, pred in observed.items()}
    # Only the four best native candidates on each prespecified objective enter the blend search.
    shortlist = sorted({
        tag for key in ("clean", "balanced")
        for tag in sorted(definitions, key=lambda name: (-scores[name][key], name))[:4]
    })
    mixtures = {tag: {tag: 1.0} for tag in names}
    for candidate in shortlist:
        for anchor in ANCHORS:
            for weight in (0.25, 0.5, 0.75):
                tag = f"{anchor}+{candidate}:{weight:.2f}"
                mixtures[tag] = {anchor: 1 - weight, candidate: weight}
    for tag, components in mixtures.items():
        if tag not in scores:
            scores[tag] = average_metrics(y, folds, {
                view: sum(weight * observed[component][view] for component, weight in components.items())
                for view in VIEWS
            })
    selected = {
        "clean_selected": max(mixtures, key=lambda key: (scores[key]["clean"], scores[key]["balanced"])),
        "balanced_selected": max(mixtures, key=lambda key: (scores[key]["balanced"], scores[key]["clean"])),
    }
    required = sorted(set().union(*(mixtures[tag].keys() for tag in selected.values())) - set(ANCHORS))
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p21-boosting-") as temporary:
        stage = Path(temporary) / OUTPUT.name
        stage.mkdir()
        final_sha = {}
        for tag in required:
            spec = definitions[tag]
            native = native_inputs(x, spec["scope"])
            shifted = native_inputs(shift(x, SHIFT_SEED + 400), spec["scope"]) if spec["augment"] else None
            model = native_fit(spec, native, y, shifted, ROUNDS[-1])
            path = stage / f"{tag}.joblib"
            joblib.dump(model, path)
            final_sha[tag] = sha(path)
            print(f"Frozen {tag}", flush=True)
        verify_sources(sources)
        result = {
            "device": "cpu", "threads": THREADS, "dataset": "P12 A/B", "algorithms": list(algorithms),
            "a_b_folds": list(FOLD_IDS), "ab_shift": SHIFT, "ab_icu_type_shift_pct": STATIC_SHIFT,
            "shift_seed": SHIFT_SEED, "clean_rank_and_balanced_rank": selected,
            "a_b_scores": scores, "selected_components": {label: mixtures[tag] for label, tag in selected.items()},
            "native_definitions": definitions, "source_sha256": sources, "model_sha256": final_sha,
            "fit_times": fit_times, "final_native_fits": len(required),
        }
        (stage / "manifest.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        stage.rename(OUTPUT)
    print(f"Frozen A/B selection: {selected}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-catboost", action="store_true")
    args = parser.parse_args()
    train(tuple(alg for alg in MODEL_PARAMS if not args.no_catboost or not alg.startswith("cat")))
