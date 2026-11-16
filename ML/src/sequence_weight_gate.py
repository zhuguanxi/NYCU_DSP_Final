"""Shared observation-hour gate for the A/B-selected sequence mixture."""

import numpy as np

GATE_CHANNELS = ("Urine", "GCS")
PROFILE_CHANNELS = ("Urine", "GCS", "MAP")


def observed_hours(cube):
    return np.isfinite(cube).sum(axis=-1)


def gated_weight(counts, groups, rule):
    urine, gcs = (counts[..., groups.index(channel)] for channel in GATE_CHANNELS)
    rich = (urine >= rule["urine_min_hours"]) & (gcs >= rule["gcs_min_hours"])
    return np.where(rich, rule["high_weight"], rule["low_weight"])
