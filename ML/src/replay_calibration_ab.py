"""Reproduce clean P33 and P38 fold scores using the existing CPU recipes."""

import sys
from pathlib import Path

import numpy as np
import torch

from sequence_blending import blend
from train_hourly_sequence import ROOT as P33, THREADS, digest, fit, read_json, validate_inputs
from train_sequence_cadence import cadence_cube
from train_sequence_channel_ablation import FITTED as P38
from select_sequence_blend import verify


def replay(destination):
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    input_path, bundle_path = P33 / "inputs/manifest.json", P33 / "inputs/ab.npz"
    p33 = read_json(P33 / "fitted/manifest.json")
    p38 = read_json(P38 / "manifest.json")
    verify(p38["input_sha256"])
    manifest = read_json(input_path)
    if (
        p33["selected"] != ["plain", 8, 0.25] or p33["input_manifest_sha256"] != digest(input_path)
        or p38["selected"] != {"ablation": "without_both", "mode": "log_odds", "weight": 0.25, "bound": 2.0}
        or manifest["views"][0] != "clean" or digest(bundle_path) != manifest["bundle_sha256"]
    ):
        raise ValueError("Unexpected original A/B sequence recipes")
    with np.load(bundle_path, allow_pickle=False) as npz:
        data = dict(npz)
    validate_inputs(data, manifest)
    if data["val_hidden"][0].any():
        raise ValueError("Expected unmasked clean A/B sequence inputs")
    np.testing.assert_array_equal(data["val_static"][0], data["static"])
    columns = [manifest["groups"].index(name) for name in p38["included_channels"]]
    reduced = {**data, "cube": cadence_cube(data["cube"])[:, columns]}
    anchor = data["anchor"][0]
    scores = {tag: np.full(anchor.shape, np.nan) for tag in ("p33", "p38")}
    for fold in range(manifest["n_folds"]):
        held = data["folds"] == fold
        for tag, source in (("p33", data), ("p38", reduced)):
            view = (torch.from_numpy(source["cube"][held]), torch.from_numpy(data["static"][held]))
            _, predictions = fit(source, ~held, fold, "plain", 8, [view])
            (scores[tag][held],) = predictions[8]
            print(f"P42 replayed {tag}, A/B fold {fold}", flush=True)
    verify(p38["input_sha256"])
    np.savez_compressed(
        destination, y=data["y"], folds=data["folds"], p28=anchor,
        p33=0.75 * anchor + 0.25 * scores["p33"], p38=blend(anchor, scores["p38"], p38["selected"]),
    )


if __name__ == "__main__":
    replay(Path(sys.argv[1]))
