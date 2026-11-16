"""Fit temporal and GCS-absent specialists, then choose a routing rule on A/B."""

import json
import tempfile
import time
from pathlib import Path

from train_predictive_boosting import (
    AB_FEATURES, FOLD_IDS, GCS_ONLY_OUTPUT, GCS_SEED, ROUNDS, SHIFT_SEED,
    ab_inputs, estimator, fit_augmented, mask_observed, native_inputs, native_predict, shift,
    source_paths, validate_oof, verified_template, verify_sources,
)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from evaluate_temporal_features import FITTED as P22_FITTED, save_json
from pilot_temporal_features import DEST as P22, temporal_view
from prepare import OUT, sha
from train_missingness_augmentation import from_json

OUTPUT = OUT / "p26_gcs_routed_temporal"
BASELINE = "gcs_only_full"
VIEWS = ("clean", "shift1", "shift2", "mapped1", "mapped2", "full_gcs")
SPECS = {
    "temporal": ("xgb3", "none", False),
    "temporal_gcs60": ("xgb3", "gcs60", False),
    "temporal_gcs85": ("xgb3", "gcs85", False),
    "temporal_shift": ("xgb3", "shift", False),
    "temporal_cat": ("cat5", "none", False),
    "blind_xgb": ("xgb3", "none", True),
    "blind_lgb": ("lgb15", "shift", True),
    "blind_cat": ("cat5", "none", True),
}
TOLERANCE = 0.005


def gcs_full(frame, seed):
    return mask_observed(frame, {"GCS": 100}, seed)[0]


def mapped_shift(frame, seed):
    masked = shift(frame, seed)
    _, chosen, _ = mask_observed(frame, {"Urine": 30}, seed + 2)
    keep = ~np.isin(np.arange(len(frame)), chosen)
    urine = [name for name in frame if name.startswith("Urine__")]
    masked.loc[masked.index[keep], urine] = frame.loc[frame.index[keep], urine]
    return masked


def build_views(frame, fold):
    seed = SHIFT_SEED + 100 * fold
    return {
        "clean": frame, "shift1": shift(frame, seed + 20), "shift2": shift(frame, seed + 30),
        "mapped1": mapped_shift(frame, seed + 20), "mapped2": mapped_shift(frame, seed + 30),
        "full_gcs": gcs_full(frame, SHIFT_SEED + fold),
    }


def matrix(raw, altered, temporal):
    return native_inputs(altered.join(temporal_view(raw, altered, temporal)), "full")


def train_one(tag, x, timing, y, seed):
    alg, aug, blind = SPECS[tag]
    base = gcs_full(x, seed) if blind else x
    native = matrix(x, base, timing)
    weight = None
    if aug != "none":
        if aug.startswith("gcs"):
            masked, changed, _ = mask_observed(base, {"GCS": int(aug[3:])}, seed)
            mass = 0.5
        else:
            masked = mapped_shift(base, seed)
            changed = np.arange(len(y))
            mass = 0.25
        shifted = matrix(x, masked, timing)[changed]
        native = np.concatenate((native, shifted))
        weight = np.ones(len(y), dtype=float)
        weight[changed] -= mass
        weight = np.r_[weight, np.full(len(changed), mass)]
        y = np.r_[y, y[changed]]
        if not np.isclose(weight.sum(), len(base)):
            raise ValueError("Augmentation altered total sample weight")
    model = estimator(alg, ROUNDS[-1])
    if alg.startswith(("xgb", "lgb")):
        model.set_params(n_jobs=2)
    model.fit(native, y, sample_weight=weight)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected outcome order")
    return model


def prediction(tag, model, values):
    return native_predict(model, {"algorithm": SPECS[tag][0]}, values, ROUNDS[-1])


def score(y, folds, probabilities):
    averages = {
        view: float(np.mean([
            average_precision_score(y[folds == fold], values[folds == fold]) for fold in FOLD_IDS
        ]))
        for view, values in probabilities.items()
    }
    averages["shift"] = (averages.pop("shift1") + averages.pop("shift2")) / 2
    averages["mapped"] = (averages.pop("mapped1") + averages.pop("mapped2")) / 2
    averages["objective"] = (
        0.55 * averages["clean"] + 0.20 * averages["mapped"]
        + 0.10 * averages["shift"] + 0.15 * averages["full_gcs"]
    )
    return averages


def candidates(y, folds, observed, routes):
    scored = {name: score(y, folds, values) for name, values in observed.items()}
    parents = [tag for tag, entry in SPECS.items() if not entry[2]]
    fallback = [tag for tag, entry in SPECS.items() if entry[2]] + [BASELINE]
    configs = {name: {"primary": name, "fallback": name, "weight": 0.0} for name in parents}
    for primary in parents:
        for specialist in fallback:
            for weight in (0.5, 1.0):
                key = f"{primary}->{specialist}:{weight:.1f}"
                config = {"primary": primary, "fallback": specialist, "weight": weight}
                configs[key] = config
                scored[key] = score(y, folds, {
                    view: np.where(
                        routes[view],
                        (1 - weight) * observed[primary][view] + weight * observed[specialist][view],
                        observed[primary][view],
                    )
                    for view in VIEWS
                })
    reference = {"clean": scored["temporal"]["clean"], "full_gcs": scored[BASELINE]["full_gcs"]}
    eligible = [
        tag for tag in configs
        if all(scored[tag][metric] >= value - TOLERANCE for metric, value in reference.items())
    ]
    winner = max(eligible or configs, key=lambda key: (scored[key]["objective"], scored[key]["clean"]))
    return scored, configs, reference, eligible, winner


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    x, y, folds = ab_inputs()
    fold_dir, inputs = source_paths()
    p22_fitted = from_json(P22_FITTED / "manifest.json")
    paths = (P22_FITTED / "manifest.json", P22_FITTED / "xgboost.joblib", *inputs)
    sources = {
        **p22_fitted["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in paths},
    }
    if sources[(P22_FITTED / "xgboost.joblib").relative_to(OUT).as_posix()] != p22_fitted["model_sha256"]:
        raise ValueError("P22 model does not match its manifest")
    verify_sources(sources)
    temporal = pd.read_parquet(P22 / "features_AB_temporal.parquet")
    if not x.index.equals(temporal.index) or list(temporal.columns) != p22_fitted["pilot"]["features"]:
        raise ValueError("A/B temporal feature keys changed")
    archived = pd.read_csv(fold_dir / "oof.csv").set_index("RecordID")
    gcs_manifest = from_json(GCS_ONLY_OUTPUT / "manifest.json")
    if (
        gcs_manifest["selected_rates_percent"] != {"GCS": 60}
        or sha(GCS_ONLY_OUTPUT / "Full.joblib") != gcs_manifest["model_sha256"]["Full"]
    ):
        raise ValueError("Expected the frozen GCS-only Full recipe")

    observed = {tag: {view: np.full(len(y), np.nan) for view in VIEWS} for tag in (*SPECS, BASELINE)}
    routes = {view: np.zeros(len(y), dtype=bool) for view in VIEWS}
    fit_times = []
    for fold in FOLD_IDS:
        train = folds != fold
        early, held = x.loc[train], x.loc[~train]
        views = build_views(held, fold)
        values = {key: matrix(held, frame, temporal.loc[~train]) for key, frame in views.items()}
        template, cols = verified_template(fold_dir / f"fold{fold}.joblib", sha(AB_FEATURES))
        validate_oof(template, cols, held, y[~train], fold, archived)
        masked, chosen, _ = mask_observed(early, {"GCS": 60}, GCS_SEED + 100 * fold)
        gcs, seconds = fit_augmented(template, cols, early, y[train], masked, chosen)
        fit_times.append({"fold": fold, "tag": BASELINE, "seconds": seconds})
        for view, frame in views.items():
            observed[BASELINE][view][~train] = gcs.predict_proba(frame.loc[:, cols])[:, 1]
            routes[view][~train] = frame["GCS__count"].eq(0).to_numpy()
        for tag in SPECS:
            started = time.perf_counter()
            model = train_one(tag, early, temporal.loc[train], y[train], SHIFT_SEED + 100 * fold)
            for view, inputs in values.items():
                observed[tag][view][~train] = prediction(tag, model, inputs)
            seconds = time.perf_counter() - started
            fit_times.append({"fold": fold, "tag": tag, "seconds": seconds})
            print(f"A/B fold {fold} {tag} in {seconds:.1f}s", flush=True)

    scores, configs, guard, eligible, winner = candidates(y, folds, observed, routes)
    for key in ("clean", "shift", "full_gcs"):
        np.testing.assert_allclose(
            scores["temporal"][key], p22_fitted["pilot"]["scores"]["temporal_650"][key], rtol=0, atol=1e-12
        )
    selected = configs[winner]
    print(f"A/B finalist {winner}; feasible choices={len(eligible)}; {scores[winner]}", flush=True)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p26-routed-") as temporary:
        stage = Path(temporary) / OUTPUT.name
        stage.mkdir()
        hashes = {}
        for tag in sorted({selected["primary"], selected["fallback"]} - {BASELINE}):
            path = stage / f"{tag}.joblib"
            if tag == "temporal":
                model = joblib.load(P22_FITTED / "xgboost.joblib")
            else:
                model = train_one(tag, x, temporal, y, SHIFT_SEED + 400)
            joblib.dump(model, path)
            hashes[tag] = sha(path)
        raw_predictions = {f"{tag}_{key}": value for tag, rows in observed.items() for key, value in rows.items()}
        np.savez_compressed(stage / "ab_candidate_predictions.npz", **raw_predictions)
        verify_sources(sources)
        save_json(stage / "manifest.json", {
            "dataset": "P12 A/B", "device": "cpu", "selected": selected, "winner": winner,
            "a_b_metrics": scores, "eligible": eligible, "guard": guard, "tolerance": TOLERANCE,
            "routing": "use fallback weight when GCS__count == 0", "fit_times": fit_times,
            "source_sha256": sources, "model_sha256": hashes,
        })
        stage.rename(OUTPUT)
    print("Frozen GCS-routed temporal pipeline", flush=True)


if __name__ == "__main__":
    run()
