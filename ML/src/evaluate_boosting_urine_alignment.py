"""Score both frozen P21 finalists under the prespecified P23 urine mapping."""

import tempfile
from pathlib import Path

from evaluate_predictive_boosting import EVALUATION, WINS, Predictor, benchmarks

import joblib
import numpy as np
import pandas as pd

from eicu_demo_features import DEMO
from evaluate_eicu_demo import source_hashes
from evaluate_eicu_urine_mapping import OUTPUT as P23, urine_events, urine_values
from evaluate_ma_gbt_p12 import eicu_inputs
from evaluate_missingness_augmentation import paired_intervals
from evaluate_temporal_features import save_json
from evaluate_temporal_urine_alignment import mapped_elastic_controls
from prepare import OUT, sha
from stress_missingness_c import METRICS, score_metrics
from train_predictive_boosting import AB_FEATURES, OUTPUT as FITTED, verify_sources
from train_missingness_augmentation import from_json

OUTPUT = OUT / "p25_eicu_boosting_urine"


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    fit_path = FITTED / "manifest.json"
    p21_path, p23_path = EVALUATION / "summary.json", P23 / "summary.json"
    fit, p21, p23 = from_json(fit_path), from_json(p21_path), from_json(p23_path)
    if p21["training"] != fit or fit["device"] != "cpu":
        raise ValueError("P21 models differ from the frozen evaluation")
    protected = {
        **p21["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in (fit_path, p21_path, p23_path)},
    }
    verify_sources(protected)
    if source_hashes(DEMO) != p23["raw_sha256"]:
        raise ValueError("P23 used different raw eICU files")
    cohort, frames = eicu_inputs()
    y = cohort.y.to_numpy()
    events, coverage = urine_events(cohort.index)
    if coverage != p23["coverage"]:
        raise ValueError("The prespecified urine mapping changed")
    urine = urine_values(cohort.index, events)
    mapped = {}
    for mode, frame in frames.items():
        update = frame.copy()
        update.loc[:, urine.columns] = urine
        mapped[mode] = update
    controls = mapped_elastic_controls(mapped, y, p23)
    models = {
        **benchmarks(),
        **{tag: joblib.load(FITTED / f"{tag}.joblib") for tag in fit["model_sha256"]},
    }
    cols = pd.read_parquet(AB_FEATURES).columns.tolist()
    prior = {(row["model"], row["mode"]): row for row in p21["eicu"]}
    results = []
    for tag in WINS:
        predictor = Predictor(fit["selected_components"][tag], models, fit["native_definitions"])
        old = {mode: predictor.predict_proba(frame.loc[:, cols])[:, 1] for mode, frame in frames.items()}
        new = {mode: predictor.predict_proba(frame.loc[:, cols])[:, 1] for mode, frame in mapped.items()}
        for mode, scores in old.items():
            original = score_metrics(y, scores)
            np.testing.assert_allclose(
                [original[key] for key in METRICS], [prior[tag, mode][key] for key in METRICS],
                atol=1e-12, rtol=0,
            )
        all_controls = {"original_alignment": old, **controls}
        intervals = {name: paired_intervals(cohort, values, new) for name, values in all_controls.items()}
        for mode, scores in new.items():
            final = score_metrics(y, scores)
            row = {"model": tag, "mode": mode, "urine_mapped": final, "comparison": {}}
            for name, values in all_controls.items():
                baseline = score_metrics(y, values[mode])
                row["comparison"][name] = {
                    "metrics": baseline, "delta": {key: final[key] - baseline[key] for key in METRICS},
                    "paired_ci": intervals[name][mode],
                }
            results.append(row)
    verify_sources(protected)
    if source_hashes(DEMO) != p23["raw_sha256"]:
        raise ValueError("Raw eICU files changed during the evaluation")
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p25-boosting-urine-") as temporary:
        staging = Path(temporary) / OUTPUT.name
        staging.mkdir()
        save_json(staging / "summary.json", {
            "device": "cpu", "fits": 0, "coverage": coverage, "rows": results,
            "source_sha256": protected, "eicu_raw_sha256": p23["raw_sha256"],
        })
        staging.rename(OUTPUT)
    print(f"Boosting and urine alignment evaluation saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    run()
