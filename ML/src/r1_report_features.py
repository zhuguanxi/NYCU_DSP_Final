"""R1 Eqs. 2–5 and adapters to the team's observation data."""

from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from prepare import UNITS, extract_record

EVENTS = ["RecordID", "elapsed_minute", "variable", "clean_value"]
CHANNELS = tuple(UNITS)
STATS = ("mean", "std", "diff_mean", "diff_std")
FEATURES = [f"{channel}__{stat}" for channel in CHANNELS for stat in STATS]


def summarize(events, index):
    if not index.is_unique:
        raise ValueError("Encounter IDs must be unique")
    if not (
        events.RecordID.isin(index).all()
        and events.variable.isin(CHANNELS).all()
        and events.elapsed_minute.between(0, 2880).all()
        and np.isfinite(events.clean_value).all()
    ):
        raise ValueError("Invalid observation, channel, timestamp, or encounter")
    names = ["RecordID", "variable"]
    raw = events.groupby(names, sort=False, observed=True).clean_value.agg(["sum", "count"])
    per_minute = events.groupby(
        names + ["elapsed_minute"], sort=False, observed=True
    ).clean_value.median().reset_index()
    per_minute.sort_values("elapsed_minute", kind="stable", inplace=True)
    per_minute["difference"] = per_minute.groupby(names, sort=False, observed=True).clean_value.diff()
    grouped = per_minute.groupby(names, sort=False, observed=True)
    statistics = pd.DataFrame({
        "mean": grouped.clean_value.mean(),
        "std": grouped.clean_value.std(ddof=0),
        "diff_mean": grouped.difference.mean(),
        "diff_std": grouped.difference.std(ddof=0),
    })
    columns = pd.MultiIndex.from_product([CHANNELS, STATS])
    features = statistics.unstack("variable").swaplevel(axis=1).reindex(index=index, columns=columns)
    features.columns = FEATURES
    features.loc[:, [col for col in FEATURES if not col.endswith("__mean")]] = (
        features.loc[:, [col for col in FEATURES if not col.endswith("__mean")]].fillna(0.0)
    )
    totals = {
        name: raw[name].unstack("variable").reindex(index=index, columns=CHANNELS).fillna(0)
        for name in ("sum", "count")
    }
    audit = {
        "raw_valid_events": len(events),
        "timestamp_events": len(per_minute),
        "same_minute_events_collapsed": len(events) - len(per_minute),
        "observed_stays_by_channel": {
            key: int(value) for key, value in totals["count"].gt(0).sum().items()
        },
    }
    return features, totals, audit


def global_means(totals, train_index):
    count = totals["count"].loc[train_index].sum(axis=0)
    if count.le(0).any():
        raise ValueError(f"No training observations for {count.index[count.le(0)].tolist()}")
    return totals["sum"].loc[train_index].sum(axis=0).div(count).to_dict()


def transform(features, means, masked=()):
    result = features.copy()
    for channel in CHANNELS:
        name = f"{channel}__mean"
        if channel in masked:
            result.loc[:, [f"{channel}__{key}" for key in STATS]] = [means[channel], 0.0, 0.0, 0.0]
        else:
            result[name] = result[name].fillna(means[channel])
    if list(result.columns) != FEATURES or not np.isfinite(result.to_numpy()).all():
        raise ValueError("R1 feature schema or imputation failed")
    return result


def p12_c_events(path, index):
    chunks = []
    seen = set()
    pending = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            stem = Path(member.name).stem
            if not member.isfile() or not stem.isdecimal():
                continue
            record = int(stem)
            if record in seen:
                raise ValueError(f"Duplicate Set C encounter {record}")
            seen.add(record)
            with archive.extractfile(member) as source:
                _, observations, _ = extract_record(source.read(), record)
            pending.extend((rid, minute, variable, value)
                           for rid, minute, variable, _, value, flag, _ in observations
                           if flag == "valid" and variable in CHANNELS)
            if len(seen) % 200 == 0:
                chunks.append(pd.DataFrame(pending, columns=EVENTS))
                pending = []
    if seen != set(index):
        raise ValueError("Set C archive and reference labels have different encounter IDs")
    if pending:
        chunks.append(pd.DataFrame(pending, columns=EVENTS))
    return pd.concat(chunks, ignore_index=True)


def check_formulas():
    index = pd.Index([1, 2, 3], name="RecordID")
    # Out of order, plus two HR measurements at minute 120 (median 4).
    events = pd.DataFrame([(1, 480, "HR", 3.0), (1, 120, "HR", 5.0),
                           (1, 0, "HR", 2.0), (1, 120, "HR", 3.0),
                           (2, 0, "HR", 8.0)], columns=EVENTS)
    features, totals, audit = summarize(events, index)
    cols = [f"HR__{stat}" for stat in STATS]
    np.testing.assert_allclose(features.loc[1, cols], [3, np.sqrt(2 / 3), 0.5, 1.5])
    np.testing.assert_array_equal(features.loc[2, cols], [8, 0, 0, 0])
    assert np.isnan(features.loc[3, "HR__mean"])
    assert (totals["count"].loc[1, "HR"], totals["sum"].loc[1, "HR"]) == (4, 13)
    assert audit["same_minute_events_collapsed"] == 1
    means = dict.fromkeys(CHANNELS, 13 / 4)
    np.testing.assert_array_equal(transform(features, means).loc[3, cols], [13 / 4, 0, 0, 0])
    np.testing.assert_array_equal(transform(features, means, ("HR",)).loc[1, cols], [13 / 4, 0, 0, 0])
