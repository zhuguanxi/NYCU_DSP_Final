"""Train CPU sequence models with per-channel eICU observation-profile augmentation."""

import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from select_sequence_gate import FITTED as P35, PROFILES, profile_view, profiles_for_fold, summaries
from train_hourly_sequence import ROOT as P33, THREADS, digest, fit, masked, predict, read_json, save_json, validate_inputs
from train_sequence_cadence import cadence_cube, tensors

ROOT = P33.parent / "p36_profile_augmentation"
FITTED = ROOT / "fitted"
SEED = 2026111736
BRANCHES = ("hourly", "four_hour")
CHECKPOINTS = (8, 16)
WEIGHTS = (0.10, 0.15, 0.25, 0.40, 1.0)


def train_copy(data, profile, groups, rows, fold, coarsen):
    base = masked(data["cube"][rows], data["aug_hidden"][fold, rows])
    augmented = profile_view(base, groups, profile, SEED + fold)
    return cadence_cube(augmented) if coarsen else augmented


def run():
    if FITTED.exists():
        raise FileExistsError(FITTED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    manifest_path, bundle_path = P33 / "inputs/manifest.json", P33 / "inputs/ab.npz"
    selection = read_json(P35)
    start = {**selection["input_sha256"], P35.relative_to(ROOT.parent).as_posix(): digest(P35)}
    for name, expected in start.items():
        if digest(ROOT.parent / name) != expected:
            raise ValueError(f"An input differs from P35: {name}")
    manifest = read_json(manifest_path)
    profile = read_json(PROFILES)
    if digest(bundle_path) != manifest["bundle_sha256"]:
        raise ValueError("Expected the original A/B sequence inputs")
    with np.load(bundle_path, allow_pickle=False) as npz:
        data = dict(npz)
    validate_inputs(data, manifest)
    y, folds = data["y"], data["folds"]
    anchor = np.concatenate((data["anchor"], data["anchor"][[manifest["views"].index(x) for x in ("mapped1", "mapped2")]]))
    coarse = {**data, "cube": cadence_cube(data["cube"])}
    options = [(tag, epoch) for tag in BRANCHES for epoch in CHECKPOINTS]
    forecasts = {key: np.full(anchor.shape, np.nan) for key in options}
    fit_times = []
    for fold in range(manifest["n_folds"]):
        test = folds == fold
        validation = profiles_for_fold(data, manifest, test, profile, fold)
        views = tuple(validation)
        for tag in BRANCHES:
            coarse_branch = tag == "four_hour"
            started = time.perf_counter()
            alternative = train_copy(data, profile, manifest["groups"], ~test, fold, coarse_branch)
            _, snapshots = fit(
                coarse if coarse_branch else data, ~test, fold, "augmented",
                CHECKPOINTS[-1], tensors(validation, coarse_branch), alternate_cube=alternative,
            )
            for epoch, predicted in snapshots.items():
                forecasts[tag, epoch][:, test] = predicted
            seconds = time.perf_counter() - started
            fit_times.append({"fold": fold, "branch": tag, "seconds": seconds})
            print(f"P36 fold {fold}, {tag}, {seconds:.1f}s", flush=True)
    if not all(np.isfinite(p).all() for p in forecasts.values()):
        raise ValueError("Missing profile-augmentation scores")
    baseline = summaries(y, folds, views, anchor)
    for key, value in baseline.items():
        np.testing.assert_allclose(value, selection["scores"]["p28"][key], atol=1e-12, rtol=0)

    scores = {key: selection["scores"][key] for key in ("p28", "p33", "p34")}
    candidates = {
        f"{tag}:{epoch}:{weight:.2f}": (tag, epoch, weight)
        for tag, epoch in options for weight in WEIGHTS
    }
    for key, (tag, epoch, weight) in candidates.items():
        scores[key] = summaries(y, folds, views, (1 - weight) * anchor + weight * forecasts[tag, epoch])
    eligible = [
        key for key in ("p33", "p34", *candidates)
        if all(scores[key][m] >= scores["p33"][m] - tol for m, tol in manifest["tolerance"].items())
    ]
    winner = max(eligible, key=lambda key: (scores[key]["objective"], scores[key]["clean"]))
    choice = candidates.get(winner)
    ROOT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="profile-fit-") as temporary:
        out = Path(temporary) / FITTED.name
        out.mkdir()
        checkpoint = None
        block = None
        if choice:
            tag, epoch, _ = choice
            coarse_branch = tag == "four_hour"
            block = 4 if coarse_branch else 1
            training = np.ones(len(y), dtype=bool)
            fold = manifest["n_folds"]
            alternative = train_copy(data, profile, manifest["groups"], training, fold, coarse_branch)
            represented = coarse if coarse_branch else data
            model, _ = fit(represented, training, fold, "augmented", epoch, (), alternate_cube=alternative)
            model_path = out / "sequence.pt"
            torch.jit.script(model).save(str(model_path))
            replay = torch.jit.load(str(model_path), map_location="cpu")
            x_demo = torch.from_numpy(represented["cube"][:16])
            s_demo = torch.from_numpy(represented["static"][:16])
            np.testing.assert_array_equal(predict(replay, x_demo, s_demo), predict(model, x_demo, s_demo))
            checkpoint = digest(model_path)
        for name, expected in start.items():
            if digest(ROOT.parent / name) != expected:
                raise ValueError(f"An input changed during fitting: {name}")
        save_json(out / "manifest.json", {
            "dataset": "P12 A/B", "device": "cpu", "threads": THREADS, "seed": SEED, "block_hours": block,
            "winner": winner, "selected": choice, "eligible": eligible, "metrics": scores,
            "a_b_fits": len(fit_times), "fit_times": fit_times, "final_fits": int(choice is not None),
            "input_sha256": start, "model_sha256": checkpoint,
        })
        out.rename(FITTED)
    print(f"P36 A/B selection: {winner}; {len(eligible)} eligible; {scores[winner]}", flush=True)


if __name__ == "__main__":
    run()
