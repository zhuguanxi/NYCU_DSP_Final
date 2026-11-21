"""Replay the original frozen controls and compare them with KEDGN transfer."""

import argparse
import json
import os
from pathlib import Path
import sys

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "src"))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_eicu_confirmation import frozen_ensemble, frozen_models
from evaluate_gcs_routed_temporal import RoutedPredictor, checked_metrics
from evaluate_hourly_sequence import infer
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import benchmarks, contrast_c, no_category_warning
from evaluate_robust_ensemble import Ensemble
from evaluate_eicu_urine_mapping import urine_values
from pilot_temporal_features import time_summary
from prepare_hourly_sequence import hourly_cube, static_values
from prepare import OUT, sha
from stress_missingness_c import load_cohort, score_metrics
from train_predictive_boosting import from_json, verify_sources

ROOT = OUT / "related_work/KEDGN/frozen_transfer_20261121"
COMMON = OUT / "related_work/common_sequence_inputs"
CONTROLS = ("frozen_full", "frozen_elastic", "gcs_only_full", "gcs_only_elastic", "p26", "p28", "p33")


def references():
    destination = ROOT / "controls.npz"
    if destination.exists():
        raise FileExistsError(destination)
    routed, _, gcs, extras, protected = frozen_models()
    rule, _, _ = frozen_ensemble(protected)
    controls = benchmarks()
    elastic, _, columns = joblib.load(OUT / "final_models/Elastic.joblib")
    elastic["model"].set_params(device="cpu", n_jobs=2)
    controls["frozen_elastic"] = elastic, columns
    prior = {
        name: from_json(OUT / f"{folder}/evaluation/summary.json")
        for name, folder in (("p26", "p26_gcs_routed_temporal"), ("p28", "p28_robust_ensemble"), ("p33", "p33_hourly_sequence"))
    }
    source_model = OUT / "p33_hourly_sequence/fitted/sequence.pt"
    if sha(source_model) != prior["p33"]["training"]["model_sha256"]:
        raise ValueError("P33 frozen sequence weight changed")
    raw, yc, history = load_cohort()
    c_events = pd.read_parquet(COMMON / "p12_c_events.parquet")
    time_c = time_summary(c_events, raw.index)
    cube_c = hourly_cube(c_events, raw.index)
    route_c = RoutedPredictor(routed.rule, routed.models, raw, time_c)
    mix_c = Ensemble(route_c, gcs, rule)
    root = ROOT / "reference_inference"
    root.mkdir(exist_ok=True)
    scores = {name: no_category_warning(pipe, raw.loc[:, cols]) for name, (pipe, cols) in controls.items()}
    scores.update({"p26": route_c.predict_proba(raw)[:, 1], "p28": mix_c.predict_proba(raw)[:, 1]})
    seq, = infer(root, source_model, cubes=cube_c[None], statics=static_values(raw)[None])
    scores["p33"] = 0.75 * scores["p28"] + 0.25 * seq.astype(float)
    for name in ("frozen_full", "frozen_elastic"):
        saved = history[f"score_{'Full' if name == 'frozen_full' else 'Elastic'}"].to_numpy(dtype=scores[name].dtype)
        np.testing.assert_array_equal(scores[name], saved)
    for name in ("p26", "p28", "p33"):
        saved, = [row for row in prior[name]["set_c"] if row["scenario"] == "clean"]
        checked_metrics(yc, scores[name], saved)
    rows = {"set_c_ids": raw.index.to_numpy(), "set_c_y": yc}
    rows.update({f"set_c_{name}": value for name, value in scores.items()})
    print("Verified original frozen Set C scores, including P26, P28 and P33.", flush=True)

    urine_prior = from_json(OUT / "p23_eicu_explicit_urine/summary.json")
    cohort = pd.read_parquet(OUT / "eicu_demo_external/cohort.parquet")
    for mode in ("hourly", "native"):
        frame = pd.read_parquet(OUT / f"eicu_demo_external/features_{mode}.parquet")
        events = pd.read_parquet(COMMON / f"eicu_{mode}_events.parquet")
        urine = events.loc[events.variable.eq("Urine")].rename(
            columns={"RecordID": "patientunitstayid", "elapsed_minute": "minute", "clean_value": "value"})
        update = urine_values(frame.index, urine)
        frame.loc[:, update.columns] = update
        timing = time_summary(events, frame.index)
        joined = frame.join(timing)
        if list(timing.columns) != extras:
            raise ValueError("External temporal columns differ")
        scores = {name: no_category_warning(pipe, frame.loc[:, cols]) for name, (pipe, cols) in controls.items()}
        route = RoutedPredictor(routed.rule, routed.models)
        scores["p26"] = route.predict_proba(joined)[:, 1]
        scores["p28"] = Ensemble(route, gcs, rule).predict_proba(joined)[:, 1]
        cube = hourly_cube(events, frame.index)
        seq, = infer(root, source_model, cubes=cube[None], statics=static_values(frame)[None])
        scores["p33"] = 0.75 * scores["p28"] + 0.25 * seq.astype(float)
        for name in ("p26", "p28", "p33"):
            saved, = [row["metrics"] for row in prior[name]["eicu"] if row["alignment"] == "urine_mapped" and row["mode"] == mode]
            checked_metrics(cohort.y.to_numpy(), scores[name], saved)
        for name, variant, model in (
            ("frozen_full", "frozen", "Full"), ("frozen_elastic", "frozen", "Elastic"),
            ("gcs_only_full", "gcs", "Full"), ("gcs_only_elastic", "gcs", "Elastic"),
        ):
            saved, = [row["urine_mapped"] for row in urine_prior["rows"]
                      if row["variant"] == variant and row["model"] == model and row["mode"] == mode]
            checked_metrics(cohort.y.to_numpy(), scores[name], saved)
        rows.update({f"eicu_{mode}_ids": cohort.index.to_numpy(), f"eicu_{mode}_y": cohort.y.to_numpy()})
        rows.update({f"eicu_{mode}_{name}": value for name, value in scores.items()})
        print(f"Verified all seven frozen {mode} eICU controls.", flush=True)
    verify_sources(protected)
    np.savez_compressed(destination, **rows)
    (ROOT / "control_sources.json").write_text(json.dumps({
        "source_sha256": protected, "controls_sha256": sha(destination),
        "p33_model_sha256": sha(source_model), "model_fits": 0,
    }, indent=2), encoding="utf-8")


def result_row(y, predictions):
    ensemble = predictions.mean(axis=0, dtype=np.float64)
    individual = [score_metrics(y, values) for values in predictions]
    metrics = score_metrics(y, ensemble)
    average = {key: {"mean": float(np.mean([row[key] for row in individual])),
                     "sd": float(np.std([row[key] for row in individual]))} for key in metrics}
    return ensemble, {"ensemble": metrics, "individual_seeds": individual, "seed_mean_sd": average}


def compare():
    output = ROOT / "summary.json"
    if output.exists():
        raise FileExistsError(output)
    frozen = from_json(ROOT / "inference.json")
    controls_manifest = from_json(ROOT / "control_sources.json")
    verify_sources(controls_manifest["source_sha256"])
    if sha(ROOT / "controls.npz") != controls_manifest["controls_sha256"]:
        raise ValueError("Frozen comparator predictions changed")
    reference = np.load(ROOT / "controls.npz")
    results = {}
    kg_eicu = {}
    for name in ("set_c", "eicu_hourly", "eicu_native"):
        path = ROOT / f"{name}_predictions.npz"
        if sha(path) != frozen["predictions"][name]["sha256"]:
            raise ValueError("KEDGN transfer prediction changed")
        with np.load(path) as predicted:
            np.testing.assert_array_equal(predicted["ids"], reference[f"{name}_ids"])
            np.testing.assert_array_equal(predicted["y"], reference[f"{name}_y"])
            probabilities, y = predicted["probabilities"], predicted["y"]
        ensemble, result = result_row(y, probabilities)
        baseline = {tag: reference[f"{name}_{tag}"] for tag in CONTROLS}
        result["controls"] = {tag: score_metrics(y, values) for tag, values in baseline.items()}
        result["delta"] = {tag: {key: result["ensemble"][key] - value for key, value in metrics.items()}
                           for tag, metrics in result["controls"].items()}
        result["stays"], result["deaths"] = len(y), int(y.sum())
        if name == "set_c":
            result["paired_ci"] = contrast_c(y, baseline, {"kedgn": ensemble})["kedgn"]
        else:
            kg_eicu[name.removeprefix("eicu_")] = ensemble
        results[name] = result
    cohort = pd.read_parquet(OUT / "eicu_demo_external/cohort.parquet")
    for tag in CONTROLS:
        bounds = paired_intervals(cohort, {mode: reference[f"eicu_{mode}_{tag}"] for mode in kg_eicu}, kg_eicu)
        for mode, bound in bounds.items():
            results[f"eicu_{mode}"].setdefault("paired_ci", {})[tag] = bound
    report = {
        "device": "cpu", "model_fits": 0, "kedgn_training": "3198 Set A stays, author split5",
        "control_training": "P12 A+B, 8000 stays", "prediction_rule": frozen["primary_prediction"],
        "results": results, "source_sha256": {
            "inference.json": sha(ROOT / "inference.json"), "control_sources.json": sha(ROOT / "control_sources.json"),
        },
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: row["ensemble"] for name, row in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("references", "compare"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with threadpool_limits(limits=2):
        {"references": references, "compare": compare}[args.phase]()
