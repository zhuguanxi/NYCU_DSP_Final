"""Select an observation-hour gate using only P12 A/B outcomes."""

import itertools
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from sequence_weight_gate import PROFILE_CHANNELS, gated_weight, observed_hours
from train_hourly_sequence import ROOT as P33, THREADS, digest, fit, read_json, save_json, validate_inputs
from train_sequence_cadence import (
    ROOT as P34, ab_scores, cadence_cube, retain_count, tensors,
)
from train_hourly_sequence import masked

ROOT = P33.parent / "p35_sequence_gate"
PROFILES = ROOT / "observation_profiles" / "summary.json"
FITTED = ROOT / "selection.json"
SEED = 2026111735
PROFILE_VIEWS = ("profile1", "profile2")
BRANCHES = ("p33", "p34")
OBJECTIVE = {"clean": 0.35, "shift": 0.10, "mapped": 0.15, "full_gcs": 0.15, "profile": 0.25}


def profile_view(cube, groups, profile, seed):
    rng = np.random.default_rng(seed)
    original = observed_hours(cube)
    limits = original.copy()
    for channel in PROFILE_CHANNELS:
        at = groups.index(channel)
        hist = np.asarray(profile["channels"][channel]["eicu"]["hour_histogram"], dtype=int)
        if hist.shape != (49,) or (hist < 0).any() or hist[1:].sum() == 0:
            raise ValueError(f"Invalid observed-hour histogram for {channel}")
        rows = np.flatnonzero(original[:, at])
        n = min(len(rows), (len(cube) * hist[1:].sum() + hist.sum() // 2) // hist.sum())
        kept = rng.permutation(rows)[:n]
        draws = rng.choice(np.arange(1, 49), size=n, p=hist[1:] / hist[1:].sum())
        limits[:, at] = 0
        limits[kept, at] = np.minimum(original[kept, at], draws)
    changed = retain_count(cube, limits, rng)
    np.testing.assert_array_equal(observed_hours(changed), limits)
    return changed


def summaries(y, folds, views, probabilities):
    original = views[:-len(PROFILE_VIEWS)]
    output = ab_scores(y, folds, original, probabilities[:len(original)])
    for i, name in enumerate(PROFILE_VIEWS, start=len(original)):
        output[name] = float(np.mean([
            average_precision_score(y[folds == fold], probabilities[i, folds == fold]) for fold in (0, 1, 2)
        ]))
    output["profile"] = sum(output[tag] for tag in PROFILE_VIEWS) / len(PROFILE_VIEWS)
    output["objective"] = sum(value * output[key] for key, value in OBJECTIVE.items())
    return output


def profiles_for_fold(data, inputs, test, profile, fold):
    original = {
        view: (masked(data["cube"][test], data["val_hidden"][i, test]), data["val_static"][i, test])
        for i, view in enumerate(inputs["views"])
    }
    for i, name in enumerate(PROFILE_VIEWS, start=1):
        cube, static = original[f"mapped{i}"]
        original[name] = profile_view(cube, inputs["groups"], profile, SEED + fold * 100 + i), static
    return original


def guard_reference(y, folds, manifest, previous, prediction):
    summary = ab_scores(y, folds, manifest["views"], prediction[:len(manifest["views"])])
    metrics = previous["metrics"][previous["winner"]]
    for key, value in summary.items():
        np.testing.assert_allclose(value, metrics[key], rtol=0, atol=1e-12)


def run():
    if FITTED.exists():
        raise FileExistsError(FITTED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    input_path, bundle_path = P33 / "inputs/manifest.json", P33 / "inputs/ab.npz"
    recipe_paths = {"p33": P33 / "fitted/manifest.json", "p34": P34 / "fitted/manifest.json"}
    model_paths = {tag: path.with_name("sequence.pt") for tag, path in recipe_paths.items()}
    paths = (input_path, bundle_path, PROFILES, *recipe_paths.values(), *model_paths.values())
    start_hashes = {path.relative_to(ROOT.parent).as_posix(): digest(path) for path in paths}
    manifest = read_json(input_path)
    if start_hashes[bundle_path.relative_to(ROOT.parent).as_posix()] != manifest["bundle_sha256"]:
        raise ValueError("Sequence bundle differs from the original A/B")
    previous = {tag: read_json(path) for tag, path in recipe_paths.items()}
    for tag, recipe in previous.items():
        if recipe["selected"] != ["plain", 8, 0.25] or recipe["device"] != "cpu":
            raise ValueError(f"Unexpected frozen {tag} recipe")
        if start_hashes[model_paths[tag].relative_to(ROOT.parent).as_posix()] != recipe["model_sha256"]:
            raise ValueError(f"Unexpected frozen {tag} checkpoint")
    profile = read_json(PROFILES)
    if profile["ab_bundle_sha256"] != manifest["bundle_sha256"] or profile["uses_outcomes_in_statistics"]:
        raise ValueError("Expected the unsupervised observed-hour profile from the same bundle")
    with np.load(bundle_path, allow_pickle=False) as npz:
        data = dict(npz)
    validate_inputs(data, manifest)
    y, folds, groups = data["y"], data["folds"], manifest["groups"]
    views = (*manifest["views"], *PROFILE_VIEWS)
    anchor = np.concatenate((data["anchor"], data["anchor"][[manifest["views"].index(x) for x in ("mapped1", "mapped2")]]))
    coarse = {**data, "cube": cadence_cube(data["cube"])}
    counts = np.zeros((len(views), len(y), len(groups)), dtype=np.int16)
    oof = {tag: np.full(anchor.shape, np.nan) for tag in BRANCHES}
    times = []
    for fold in range(manifest["n_folds"]):
        test = folds == fold
        fold_input = profiles_for_fold(data, manifest, test, profile, fold)
        for i, (cube, _) in enumerate(fold_input.values()):
            counts[i, test] = observed_hours(cube)
        for tag in BRANCHES:
            started = time.perf_counter()
            represented = coarse if tag == "p34" else data
            _, predictions = fit(represented, ~test, fold, "plain", 8, tensors(fold_input, tag == "p34"))
            oof[tag][:, test] = predictions[8]
            duration = time.perf_counter() - started
            times.append({"fold": fold, "branch": tag, "seconds": duration})
            print(f"Replayed {tag} A/B fold {fold}, {duration:.1f}s", flush=True)
    if not all(np.isfinite(p).all() for p in oof.values()):
        raise ValueError("Incomplete sequence branch folds")

    scores = {"p28": summaries(y, folds, views, anchor)}
    fixed = {}
    for tag in BRANCHES:
        fixed[tag] = 0.75 * anchor + 0.25 * oof[tag]
        guard_reference(y, folds, manifest, previous[tag], fixed[tag])
        scores[tag] = summaries(y, folds, views, fixed[tag])
    choices = {}
    gate_counts = {}
    for tag, urine, gcs, low, high in itertools.product(
        BRANCHES, (8, 16, 24), (0, 6, 10), (0.0, 0.10), (0.25, 0.40),
    ):
        key = f"{tag}:urine{urine}:gcs{gcs}:{low:.2f}:{high:.2f}"
        rule = {"branch": tag, "urine_min_hours": urine, "gcs_min_hours": gcs, "low_weight": low, "high_weight": high}
        weight = gated_weight(counts, groups, rule)
        scores[key] = summaries(y, folds, views, (1 - weight) * anchor + weight * oof[tag])
        choices[key] = rule
        gate_counts[key] = {view: int((weight[i] == high).sum()) for i, view in enumerate(views)}
    eligible = [
        name for name in (*BRANCHES, *choices)
        if all(scores[name][m] >= scores["p33"][m] - tolerance for m, tolerance in manifest["tolerance"].items())
    ]
    winner = max(eligible, key=lambda key: (scores[key]["objective"], scores[key]["clean"]))
    selected = choices.get(winner)
    for path in paths:
        if digest(path) != start_hashes[path.relative_to(ROOT.parent).as_posix()]:
            raise ValueError(f"An input changed while selecting the gate: {path.name}")
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="gate-") as temporary:
        output = Path(temporary) / FITTED.name
        save_json(output, {
            "dataset": "P12 A/B", "device": "cpu", "seed": SEED, "winner": winner,
            "selected": selected, "scores": scores, "eligible": eligible, "fit_times": times,
            "final_fits": 0, "gated_high_stays": gate_counts.get(winner), "groups": groups,
            "input_sha256": start_hashes, "objective_weights": OBJECTIVE,
            "profile_counts": {
                view: {channel: {
                    "observed_stays": int((counts[i, :, groups.index(channel)] > 0).sum()),
                    "observed_hours": int(counts[i, :, groups.index(channel)].sum()),
                } for channel in PROFILE_CHANNELS}
                for i, view in enumerate(views) if view.startswith(("mapped", "profile"))
            },
        })
        output.rename(FITTED)
    print(f"P35 A/B selection: {winner}; {len(eligible)} eligible; {scores[winner]}", flush=True)


if __name__ == "__main__":
    run()
