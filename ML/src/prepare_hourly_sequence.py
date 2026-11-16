"""Prepare 48-hour data and P28 reference scores for the CPU sequence trial."""

import tempfile
from pathlib import Path

from evaluate_monitoring_ablation import previous_scores

import numpy as np
import pandas as pd

from evaluate_temporal_features import save_json
from pilot_temporal_features import GROUPS, ID, MINUTE, OBSERVATIONS, VALUE
from prepare import OUT, sha
from select_robust_ensemble import OBJECTIVE, OUTPUT as P28, TOLERANCE
from train_gcs_routed_temporal import OUTPUT as P26, VIEWS, build_views, mapped_shift
from train_predictive_boosting import SHIFT_SEED, ab_inputs, from_json, verify_sources

ROOT = OUT / "p33_hourly_sequence"
PREPARED = ROOT / "inputs"
HOURS = 48
STATIC_NUM = ("static_Age", "static_Height", "static_Weight")
STATIC_CODES = {"static_Gender": (0, 1), "static_ICUType": (1, 2, 3, 4)}
REFIT_ID = 3


def hourly_cube(records, index):
    """Place the hourly medians at their encounter, variable and hour indices."""
    events = records.loc[
        records["variable"].isin(GROUPS) & records[MINUTE].between(0, 2880) & records[VALUE].notna(),
        [ID, MINUTE, "variable", VALUE],
    ].copy()
    events["hour"] = (events[MINUTE] // 60).clip(upper=HOURS - 1).astype(int)
    hours = events.groupby([ID, "variable", "hour"], as_index=False, sort=False)[VALUE].median()
    row = index.get_indexer(hours[ID])
    channel = pd.Index(GROUPS).get_indexer(hours["variable"])
    values = hours[VALUE].to_numpy(dtype=np.float32)
    if not index.is_unique or (row < 0).any() or (channel < 0).any() or not np.isfinite(values).all():
        raise ValueError("Hourly inputs include an invalid key or value")
    cube = np.full((len(index), len(GROUPS), HOURS), np.nan, dtype=np.float32)
    cube[row, channel, hours["hour"]] = values
    return cube


def static_values(frame):
    continuous = frame.loc[:, STATIC_NUM].to_numpy(dtype=np.float32)
    result = [continuous]
    for key, levels in STATIC_CODES.items():
        value = frame[key].to_numpy(dtype=np.float32)[:, None]
        result.append(np.where(np.isnan(value), np.nan, value == levels).astype(np.float32))
    return np.column_stack(result)


def hidden_channels(original, masked):
    counts = [f"{channel}__count" for channel in GROUPS]
    return original.loc[:, counts].gt(0).to_numpy() & masked.loc[:, counts].eq(0).to_numpy()


def prepare_data():
    if PREPARED.exists():
        raise FileExistsError(PREPARED)
    paths = [P28 / "manifest.json", P26 / "manifest.json", OBSERVATIONS]
    selection, route = map(from_json, paths[:2])
    protected = {**selection["source_sha256"], **{path.relative_to(OUT).as_posix(): sha(path) for path in paths}}
    verify_sources(protected)
    raw, y, folds = ab_inputs()
    records = pd.read_parquet(OBSERVATIONS, columns=[ID, MINUTE, "variable", VALUE, "quality_flag"])
    cube = hourly_cube(records.loc[records.quality_flag.eq("valid")], raw.index)
    static = static_values(raw)
    present = ~np.isnan(cube).all(axis=2)
    counts = [f"{channel}__count" for channel in GROUPS]
    np.testing.assert_array_equal(present, raw.loc[:, counts].gt(0).to_numpy())
    previous = previous_scores(raw, y, folds, selection, route)

    val_hidden = np.zeros((len(VIEWS), *present.shape), dtype=bool)
    val_static = np.full((len(VIEWS), *static.shape), np.nan, dtype=np.float32)
    aug_hidden = np.zeros((REFIT_ID + 1, *present.shape), dtype=bool)
    aug_static = np.full((REFIT_ID + 1, *static.shape), np.nan, dtype=np.float32)
    for fold in range(REFIT_ID):
        held = folds == fold
        views = build_views(raw.loc[held], fold)
        for i, view in enumerate(VIEWS):
            val_hidden[i, held] = hidden_channels(raw.loc[held], views[view])
            val_static[i, held] = static_values(views[view])
    for fold in range(REFIT_ID + 1):
        train = np.ones(len(y), dtype=bool) if fold == REFIT_ID else folds != fold
        seed = SHIFT_SEED + (400 if fold == REFIT_ID else 100 * fold)
        early = raw.loc[train]
        shifted = mapped_shift(early, seed)
        aug_hidden[fold, train] = hidden_channels(early, shifted)
        aug_static[fold, train] = static_values(shifted)

    ROOT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="inputs-") as temporary:
        staging = Path(temporary) / PREPARED.name
        staging.mkdir()
        bundle = staging / "ab.npz"
        np.savez_compressed(
            bundle, y=y, folds=folds, cube=cube, static=static, val_hidden=val_hidden, val_static=val_static,
            aug_hidden=aug_hidden, aug_static=aug_static, anchor=np.stack([previous[view] for view in VIEWS]),
        )
        verify_sources(protected)
        save_json(staging / "manifest.json", {
            "dataset": "P12 A/B", "device": "cpu", "views": VIEWS, "groups": GROUPS, "hours": HOURS,
            "static_columns": STATIC_NUM, "static_codes": STATIC_CODES, "n_folds": REFIT_ID,
            "stays": len(y), "deaths": int(y.sum()), "anchor": selection["a_b_metrics"][selection["winner"]],
            "weights": OBJECTIVE, "tolerance": TOLERANCE, "source_sha256": protected, "bundle_sha256": sha(bundle),
        })
        staging.rename(PREPARED)
    print("Prepared 48 hourly bins, 20 channels and exact P28 A/B scores", flush=True)


if __name__ == "__main__":
    prepare_data()
