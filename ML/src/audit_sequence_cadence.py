"""Measure selected observed-hour differences on the existing P12 and eICU cohorts."""

import argparse
import tempfile
from pathlib import Path

from prepare_hourly_sequence import GROUPS, ID, MINUTE, PREPARED, VALUE, hourly_cube

import numpy as np
import pandas as pd

from eicu_demo_features import DEMO, STAY, monitor_events, nurse_events, select_cohort
from evaluate_eicu_demo import source_hashes
from evaluate_eicu_urine_mapping import OUTPUT as URINE, urine_events
from evaluate_temporal_features import save_json
from prepare import OUT, sha
from train_predictive_boosting import from_json, verify_sources

OUTPUT = OUT / "p34_sequence_cadence" / "observation_audit"
MONITOR = {"heartrate": "HR", "respiration": "RespRate", "systemicmean": "MAP"}
CHANNELS = ("HR", "RespRate", "MAP", "GCS", "Urine")


def counts(cube, channel, histogram=False):
    present = np.isfinite(cube[:, GROUPS.index(channel)])
    total = present.sum(axis=1)
    observed = total > 0
    summary = {
        "stays": len(total), "observed_stays": int(observed.sum()),
        "median_hours_among_observed": float(np.median(total[observed])) if observed.any() else None,
        "no_last_12h_among_observed": int((observed & ~present[:, 36:].any(axis=1)).sum()),
    }
    if histogram:
        summary["hour_histogram"] = np.bincount(total, minlength=49).tolist()
    return summary


def run(histograms=False):
    output = OUT / "p35_sequence_gate" / "observation_profiles" if histograms else OUTPUT
    if output.exists():
        raise FileExistsError(output)
    manifest = from_json(PREPARED / "manifest.json")
    bundle = PREPARED / "ab.npz"
    verify_sources(manifest["source_sha256"])
    if sha(bundle) != manifest["bundle_sha256"]:
        raise ValueError("A/B inputs differ from P33")
    expected = from_json(URINE / "summary.json")
    raw = source_hashes(DEMO)
    if raw != expected["raw_sha256"]:
        raise ValueError("Demo inputs changed")
    with np.load(bundle, allow_pickle=False) as npz:
        ab = npz["cube"]
    cohort, _, _ = select_cohort(DEMO)
    monitor = monitor_events(DEMO, "vitalPeriodic", cohort.index, MONITOR)
    gcs, _, _ = nurse_events(DEMO, cohort.index)
    urine, urine_coverage = urine_events(cohort.index)
    if urine_coverage != expected["coverage"]:
        raise ValueError("Urine inputs differ from previous alignment")
    events = pd.concat((monitor, gcs, urine), ignore_index=True)
    demo = hourly_cube(events.rename(columns={STAY: ID, "minute": MINUTE, "value": VALUE}), cohort.index)
    original = pd.read_csv(OUT / "eicu_demo_external/feature_coverage.csv")
    observed = original.loc[original.cohort.eq("native")].set_index("variable")["observed_n"]
    result = {}
    for channel in CHANNELS:
        row = {"ab": counts(ab, channel, histogram=histograms), "eicu": counts(demo, channel, histogram=histograms)}
        expected_count = urine_coverage["observed_stays"] if channel == "Urine" else observed[channel]
        if row["eicu"]["observed_stays"] != expected_count:
            raise ValueError(f"Coverage changed for {channel}")
        result[channel] = row
    if source_hashes(DEMO) != raw or sha(bundle) != manifest["bundle_sha256"]:
        raise ValueError("An input changed during the audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="observation-audit-") as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        save_json(stage / "summary.json", {
            "channels": result, "device": "cpu", "uses_outcomes_in_statistics": False,
            "ab_bundle_sha256": manifest["bundle_sha256"], "eicu_raw_sha256": raw,
        })
        stage.rename(output)
    print("Verified five P12 and eICU observation-hour comparisons", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histograms", action="store_true")
    run(histograms=parser.parse_args().histograms)
