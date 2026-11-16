"""Combine fixed sequence and tabular probabilities on a declared scale."""

import numpy as np
from scipy.special import expit, logit


def blend(anchor, sequence, rule):
    anchor = np.asarray(anchor, dtype=np.float64)
    sequence = np.asarray(sequence, dtype=np.float64)
    weight = rule["weight"]
    if rule["mode"] == "probability":
        return (1 - weight) * anchor + weight * sequence
    if rule["mode"] != "log_odds":
        raise ValueError(f"Unknown blend scale: {rule['mode']}")
    base = logit(np.clip(anchor, 1e-7, 1 - 1e-7))
    correction = logit(np.clip(sequence, 1e-7, 1 - 1e-7)) - base
    bound = rule["bound"]
    if bound is not None:
        correction = np.clip(correction, -bound, bound)
    return expit(base + weight * correction)
