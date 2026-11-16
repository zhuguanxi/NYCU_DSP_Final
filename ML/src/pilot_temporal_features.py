"""A/B-only pilot of hourly time-window features versus identical native XGBoost."""

import json
import tempfile
import time
from pathlib import Path

from train_predictive_boosting import (
    AB_FEATURES, AB_FOLDS, AB_LABELS, FOLD_IDS, ROUNDS, SHIFT_SEED, VIEWS,
    ab_inputs, average_metrics, estimator, mask_observed, native_inputs, native_predict, shift, verify_sources,
)

import numpy as np
import pandas as pd

from prepare import OUT, sha

DEST = OUT / "p22_temporal_ab_pilot"
OBSERVATIONS = OUT / "observations_AB.parquet"
GROUPS = (
    "GCS", "Urine", "BUN", "Creatinine", "Lactate", "Bilirubin", "HCO3", "pH",
    "HR", "RespRate", "Temp", "SysABP", "NISysABP", "MAP", "NIMAP",
    "FiO2", "SaO2", "PaO2", "WBC", "Platelets",
)
ID, MINUTE, VALUE = "RecordID", "elapsed_minute", "clean_value"
STATS = ("early_mean", "late_mean", "late_min", "late_max", "late_minus_early", "hours_since_last", "covered_hours")


def time_summary(records, index):
    """Build a fixed schema with equal weight for each observed hour."""
    rows = records.loc[
        records["variable"].isin(GROUPS) & records[MINUTE].between(0, 2880) & records[VALUE].notna(),
        [ID, "variable", MINUTE, VALUE],
    ]
    if not pd.Index(rows[ID].unique()).isin(index).all():
        raise ValueError("Temporal rows include an ID outside the intended cohort")
    if not np.isfinite(rows[VALUE].to_numpy()).all():
        raise ValueError("Nonfinite entries in the accepted observations")
    hours = rows.assign(hour=(rows[MINUTE] // 60).clip(upper=47)).groupby(
        [ID, "variable", "hour"], sort=False, observed=True
    )[VALUE].median().rename("value").reset_index()
    early = hours.loc[hours.hour.lt(12)].groupby([ID, "variable"], sort=False, observed=True).value.mean()
    late = hours.loc[hours.hour.ge(36)].groupby([ID, "variable"], sort=False, observed=True).value
    full = hours.groupby([ID, "variable"], sort=False, observed=True)
    age = (2880 - rows.groupby([ID, "variable"], sort=False, observed=True)[MINUTE].max()) / 60
    measures = {
        "early_mean": early, "late_mean": late.mean(), "late_min": late.min(), "late_max": late.max(),
        "hours_since_last": age, "covered_hours": full.size(),
    }
    matrices = {name: s.unstack("variable").reindex(index=index, columns=GROUPS) for name, s in measures.items()}
    matrices["late_minus_early"] = matrices["late_mean"] - matrices["early_mean"]
    matrices["covered_hours"] = matrices["covered_hours"].fillna(0)
    return pd.DataFrame(
        {f"temporal_{group}__{stat}": matrices[stat][group] for group in GROUPS for stat in STATS}, index=index,
    )


def temporal_view(clean, masked, temporal):
    output = temporal.copy()
    for group in GROUPS:
        hidden = clean[f"{group}__count"].gt(0) & masked[f"{group}__count"].eq(0)
        if hidden.any():
            columns = [f"temporal_{group}__{stat}" for stat in STATS if stat != "covered_hours"]
            output.loc[hidden, columns] = np.nan
            output.loc[hidden, f"temporal_{group}__covered_hours"] = 0
    return output


def run():
    if DEST.exists():
        raise FileExistsError(DEST)
    inputs = (AB_FEATURES, AB_FOLDS, AB_LABELS, OBSERVATIONS)
    sources = {path.relative_to(OUT).as_posix(): sha(path) for path in inputs}
    x, y, folds = ab_inputs()
    records = pd.read_parquet(OBSERVATIONS, columns=[ID, MINUTE, VALUE, "variable", "quality_flag"])
    temporal = time_summary(records.loc[records.quality_flag.eq("valid")], x.index)
    if temporal.shape != (8000, len(GROUPS) * len(STATS)) or not temporal.index.equals(x.index):
        raise ValueError("Temporal features differ from the A/B cohort")
    if not temporal.filter(like="__covered_hours").apply(lambda values: values.between(0, 48)).all().all():
        raise ValueError("Temporal hour counts exceed the observation window")
    predictions = {
        f"{tag}_{n}": {v: np.full(len(x), np.nan) for v in VIEWS}
        for tag in ("original", "temporal") for n in ROUNDS
    }
    spec = {"algorithm": "xgb3", "scope": "full", "augment": False}
    for fold in FOLD_IDS:
        trained = folds != fold
        held = x.loc[~trained]
        views = {
            "clean": held, "shift1": shift(held, SHIFT_SEED + 100 * fold + 20),
            "shift2": shift(held, SHIFT_SEED + 100 * fold + 30),
            "full_gcs": mask_observed(held, {"GCS": 100}, SHIFT_SEED + fold)[0],
        }
        for tag in ("original", "temporal"):
            train_x = x.loc[trained]
            if tag == "temporal":
                train_x = train_x.join(temporal)
                eval_frames = {key: f.join(temporal_view(held, f, temporal.loc[~trained])) for key, f in views.items()}
            else:
                eval_frames = views
            model = estimator("xgb3", ROUNDS[-1]).set_params(n_jobs=2)
            start = time.perf_counter()
            model.fit(native_inputs(train_x, "full"), y[trained])
            for key, frame in eval_frames.items():
                mat = native_inputs(frame, "full")
                for n in ROUNDS:
                    predictions[f"{tag}_{n}"][key][~trained] = native_predict(model, spec, mat, n)
            print(f"A/B {tag} fold {fold} in {time.perf_counter() - start:.1f}s", flush=True)
    scores = {tag: average_metrics(y, folds, values) for tag, values in predictions.items()}
    verify_sources(sources)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p22-temporal-") as temporary:
        staging = Path(temporary) / DEST.name
        staging.mkdir()
        temporal.to_parquet(staging / "features_AB_temporal.parquet")
        report = {
            "dataset": "P12 A/B", "shape": list(temporal.shape), "features": list(temporal.columns),
            "device": "cpu", "threads": 2, "scores": scores, "fit_count": 6, "source_sha256": sources,
        }
        (staging / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(DEST)
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    run()
