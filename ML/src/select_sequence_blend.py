"""Compare fixed combination rules using the original P12 A/B folds."""

import itertools
import tempfile
from pathlib import Path

import numpy as np
import torch

from select_sequence_gate import (
    BRANCHES, FITTED as P35, OBJECTIVE, PROFILE_VIEWS, PROFILES, profiles_for_fold, summaries,
)
from sequence_blending import blend
from train_hourly_sequence import ROOT as P33, THREADS, digest, fit, read_json, save_json, validate_inputs
from train_sequence_cadence import cadence_cube, tensors

ROOT = P33.parent / "p37_sequence_blend"
SELECTION = ROOT / "selection.json"
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.40)
BOUNDS = (0.5, 1.0, 2.0, None)


def candidates():
    options = {}
    for tag, weight in itertools.product(BRANCHES, WEIGHTS):
        if weight != 0.25:
            options[f"{tag}:probability:{weight:.2f}"] = {
                "branch": tag, "mode": "probability", "weight": weight, "bound": None,
            }
        for bound in BOUNDS:
            key = f"{tag}:log_odds:{weight:.2f}:{bound if bound is not None else 'unbounded'}"
            options[key] = {"branch": tag, "mode": "log_odds", "weight": weight, "bound": bound}
    return options


def verify(hashes):
    for name, expected in hashes.items():
        if digest(ROOT.parent / name) != expected:
            raise ValueError(f"A frozen input differs from P35: {name}")


def run():
    if SELECTION.exists():
        raise FileExistsError(SELECTION)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    previous = read_json(P35)
    hashes = {**previous["input_sha256"], P35.relative_to(ROOT.parent).as_posix(): digest(P35)}
    verify(hashes)
    manifest = read_json(P33 / "inputs/manifest.json")
    profile = read_json(PROFILES)
    with np.load(P33 / "inputs/ab.npz", allow_pickle=False) as bundle:
        data = dict(bundle)
    validate_inputs(data, manifest)
    y, folds = data["y"], data["folds"]
    views = (*manifest["views"], *PROFILE_VIEWS)
    profile_anchors = [manifest["views"].index(name) for name in ("mapped1", "mapped2")]
    anchor = np.concatenate((data["anchor"], data["anchor"][profile_anchors]))
    coarse = {**data, "cube": cadence_cube(data["cube"])}
    forecasts = {tag: np.full(anchor.shape, np.nan) for tag in BRANCHES}
    fits = []
    for fold in range(manifest["n_folds"]):
        test = folds == fold
        validation = profiles_for_fold(data, manifest, test, profile, fold)
        if tuple(validation) != views:
            raise ValueError("A/B validation views changed")
        for tag in BRANCHES:
            coarsen = tag == "p34"
            _, scores = fit(coarse if coarsen else data, ~test, fold, "plain", 8, tensors(validation, coarsen))
            forecasts[tag][:, test] = scores[8]
            fits.append({"fold": fold, "branch": tag})
            print(f"P37 replayed {tag} A/B fold {fold}", flush=True)
    if not all((np.isfinite(p) & (p >= 0) & (p <= 1)).all() for p in forecasts.values()):
        raise ValueError("Invalid or incomplete A/B sequence probabilities")

    metrics = {"p28": summaries(y, folds, views, anchor)}
    for tag in BRANCHES:
        fixed = blend(anchor, forecasts[tag], {"mode": "probability", "weight": 0.25})
        metrics[tag] = summaries(y, folds, views, fixed)
    for key, scores in metrics.items():
        for name, score in scores.items():
            np.testing.assert_allclose(score, previous["scores"][key][name], rtol=0, atol=1e-12)
    options = candidates()
    for key, rule in options.items():
        metrics[key] = summaries(y, folds, views, blend(anchor, forecasts[rule["branch"]], rule))
    eligible = [
        key for key in (*BRANCHES, *options)
        if all(metrics[key][name] >= metrics["p33"][name] - tol for name, tol in manifest["tolerance"].items())
    ]
    winner = max(eligible, key=lambda key: (metrics[key]["objective"], metrics[key]["clean"]))
    verify(hashes)
    ROOT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="selection-") as temporary:
        output = Path(temporary) / SELECTION.name
        save_json(output, {
            "dataset": "P12 A/B", "device": "cpu", "threads": THREADS, "winner": winner,
            "selected": options.get(winner), "candidates": options, "eligible": eligible, "scores": metrics,
            "a_b_fits": fits, "final_fits": 0, "objective_weights": OBJECTIVE,
            "tolerance": manifest["tolerance"], "input_sha256": hashes,
        })
        output.rename(SELECTION)
    print(f"P37 A/B selected {winner} from {len(eligible)} eligible rules: {metrics[winner]}", flush=True)


if __name__ == "__main__":
    run()
