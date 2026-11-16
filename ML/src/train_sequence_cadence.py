"""Compare a common four-hour cadence and time-missingness augmentation on P12 A/B."""

import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from train_hourly_sequence import ROOT as P33, THREADS, digest, fit, masked, predict, read_json, save_json, validate_inputs

ROOT = P33.parent / "p34_sequence_cadence"
FITTED = ROOT / "fitted"
BLOCK = 4
LATE_START = 36
SEED = 2026111734
STRESS = ("random_thin", "random_same_count", "late_outage")
CHECKPOINTS = (8, 16)
WEIGHTS = (0.10, 0.15, 0.25, 0.50, 1.0)
OBJECTIVE = {"clean": 0.35, "shift": 0.10, "mapped": 0.15, "full_gcs": 0.15, "stress": 0.25}


def cadence_cube(cube):
    """Keep a single median at the end of each fixed four-hour block."""
    if cube.ndim != 3 or cube.shape[-1] != 48:
        raise ValueError("Expected encounter, channel and 48 hourly values")
    blocks = cube.reshape((*cube.shape[:2], -1, BLOCK))
    observed = np.isfinite(blocks).any(axis=-1)
    block_value = np.full(observed.shape, np.nan, dtype=np.float32)
    block_value[observed] = np.nanmedian(blocks[observed], axis=-1)
    result = np.full_like(cube, np.nan)
    result[:, :, BLOCK - 1::BLOCK] = block_value
    return result


def retain_count(cube, keep, rng):
    observed = np.isfinite(cube)
    if keep.shape != cube.shape[:2] or (keep > observed.sum(axis=2)).any() or (keep < 0).any():
        raise ValueError("Unexpected observed-hour quota")
    priority = np.where(observed, rng.random(cube.shape, dtype=np.float32), np.inf)
    ordered = np.argsort(priority, axis=2, kind="stable")
    retain = np.zeros(cube.shape, dtype=bool)
    valid = np.broadcast_to(np.arange(cube.shape[2]), cube.shape) < keep[:, :, None]
    np.put_along_axis(retain, ordered, valid, axis=2)
    return np.where(retain, cube, np.nan)


def stress_inputs(mapped, seed):
    rng = np.random.default_rng(seed)
    count = np.isfinite(mapped).sum(axis=2)
    late = mapped.copy()
    late[:, :, LATE_START:] = np.nan
    matched = np.isfinite(late).sum(axis=2)
    random_same_count = retain_count(mapped, matched, rng)
    np.testing.assert_array_equal(np.isfinite(random_same_count).sum(axis=2), matched)
    return {
        "random_thin": retain_count(mapped, (count + 1) // 2, rng),
        "random_same_count": random_same_count,
        "late_outage": late,
    }


def train_alternate(data, training, fold):
    base = masked(data["cube"][training], data["aug_hidden"][fold, training])
    rng = np.random.default_rng(SEED + 500 + fold)
    count = np.isfinite(base).sum(axis=2)
    revised = retain_count(base, (count + 1) // 2, rng)
    late = rng.permutation(len(base))[:len(base) // 2]
    revised[late, :, :] = base[late]
    revised[late, :, LATE_START:] = np.nan
    return cadence_cube(revised)


def ab_scores(y, folds, views, probabilities):
    values = {
        view: float(np.mean([average_precision_score(y[folds == k], p[folds == k]) for k in (0, 1, 2)]))
        for view, p in zip(views, probabilities, strict=True)
    }
    values["shift"] = (values.pop("shift1") + values.pop("shift2")) / 2
    values["mapped"] = (values.pop("mapped1") + values.pop("mapped2")) / 2
    if any(view in values for view in STRESS):
        values["stress"] = float(np.mean([values[tag] for tag in STRESS]))
        values["objective"] = sum(amount * values[tag] for tag, amount in OBJECTIVE.items())
    return values


def fold_views(data, original_views, test, fold):
    values = {
        name: (masked(data["cube"][test], data["val_hidden"][i, test]), data["val_static"][i, test])
        for i, name in enumerate(original_views)
    }
    cube, static = values["mapped1"]
    stress = stress_inputs(cube, SEED + fold)
    values.update({name: (changed, static) for name, changed in stress.items()})
    return values


def tensors(views, coarsen):
    return [
        (torch.from_numpy(cadence_cube(cube) if coarsen else cube), torch.from_numpy(static))
        for cube, static in views.values()
    ]


def selection():
    if FITTED.exists():
        raise FileExistsError(FITTED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    input_path = P33 / "inputs/manifest.json"
    bundle_path = P33 / "inputs/ab.npz"
    p33_path = P33 / "fitted/manifest.json"
    inputs = (input_path, bundle_path, p33_path)
    input_hashes = {path.relative_to(ROOT.parent).as_posix(): digest(path) for path in inputs}
    manifest, previous = read_json(input_path), read_json(p33_path)
    if digest(bundle_path) != manifest["bundle_sha256"] or digest(input_path) != previous["input_manifest_sha256"]:
        raise ValueError("The P33 A/B bundle changed")
    if previous["selected"] != ["plain", 8, 0.25] or previous["device"] != "cpu":
        raise ValueError("Expected the reported P33 CPU recipe")
    with np.load(bundle_path, allow_pickle=False) as archive:
        data = dict(archive)
    validate_inputs(data, manifest)
    original_views = manifest["views"]
    views = (*original_views, *STRESS)
    y, folds = data["y"], data["folds"]
    anchor = np.concatenate((data["anchor"], np.repeat(data["anchor"][[original_views.index("mapped1")]], len(STRESS), axis=0)))
    coarse = {**data, "cube": cadence_cube(data["cube"])}
    trial = [(tag, epoch) for tag in ("plain", "augmented") for epoch in CHECKPOINTS]
    predictions = {key: np.full(anchor.shape, np.nan) for key in (("p33", 8), *trial)}
    times = []
    stress_counts = {name: 0 for name in ("mapped1", *STRESS)}
    for fold in range(manifest["n_folds"]):
        test = folds == fold
        validation = fold_views(data, original_views, test, fold)
        for name in stress_counts:
            stress_counts[name] += int(np.isfinite(validation[name][0]).sum())
        start = time.perf_counter()
        _, replay = fit(data, ~test, fold, "plain", 8, tensors(validation, False))
        predictions["p33", 8][:, test] = replay[8]
        times.append({"fold": fold, "type": "p33_replay", "seconds": time.perf_counter() - start})
        coarse_views = tensors(validation, True)
        alternate = train_alternate(data, ~test, fold)
        for policy in ("plain", "augmented"):
            start = time.perf_counter()
            _, outputs = fit(
                coarse, ~test, fold, policy, CHECKPOINTS[-1], coarse_views,
                alternate_cube=alternate if policy == "augmented" else None,
            )
            for epoch, proposed in outputs.items():
                predictions[policy, epoch][:, test] = proposed
            elapsed = time.perf_counter() - start
            times.append({"fold": fold, "type": policy, "seconds": elapsed})
            print(f"Finished P34 A/B fold {fold}, {policy}, {elapsed:.1f}s", flush=True)

    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Incomplete A/B cadence scores")
    if stress_counts["random_same_count"] != stress_counts["late_outage"]:
        raise ValueError("Late outage and random views must retain identical observed-hour counts")
    p33 = 0.75 * anchor + 0.25 * predictions["p33", 8]
    measures = ab_scores(y, folds, original_views, p33[:len(original_views)])
    for key, actual in measures.items():
        np.testing.assert_allclose(actual, previous["metrics"][previous["winner"]][key], atol=1e-12, rtol=0)
    scores = {"p28": ab_scores(y, folds, views, anchor), "p33": ab_scores(y, folds, views, p33)}
    options = {
        f"{policy}:{epoch}:{weight:.2f}": (policy, epoch, weight)
        for policy, epoch in trial for weight in WEIGHTS
    }
    for name, (policy, epoch, weight) in options.items():
        proposed = weight * predictions[policy, epoch] + (1 - weight) * anchor
        scores[name] = ab_scores(y, folds, views, proposed)
    eligible = [
        name for name in ("p33", *options)
        if all(scores[name][m] >= scores["p33"][m] - tolerance for m, tolerance in manifest["tolerance"].items())
    ]
    winner = max(eligible, key=lambda name: (scores[name]["objective"], scores[name]["clean"]))
    choice = options.get(winner)
    ROOT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="cadence-fit-") as temporary:
        stage = Path(temporary) / FITTED.name
        stage.mkdir()
        checkpoint = None
        if choice:
            policy, epoch, _ = choice
            train = np.ones(len(y), dtype=bool)
            refit = manifest["n_folds"]
            alternate = train_alternate(data, train, refit) if policy == "augmented" else None
            model, _ = fit(coarse, train, refit, policy, epoch, (), alternate_cube=alternate)
            model_path = stage / "sequence.pt"
            torch.jit.script(model).save(str(model_path))
            replay = torch.jit.load(str(model_path), map_location="cpu")
            x_demo, s_demo = torch.from_numpy(coarse["cube"][:16]), torch.from_numpy(coarse["static"][:16])
            np.testing.assert_array_equal(predict(replay, x_demo, s_demo), predict(model, x_demo, s_demo))
            checkpoint = digest(model_path)
        for path in inputs:
            if digest(path) != input_hashes[path.relative_to(ROOT.parent).as_posix()]:
                raise ValueError(f"The A/B input changed during fitting: {path.name}")
        save_json(stage / "manifest.json", {
            "dataset": "P12 A/B", "device": "cpu", "threads": THREADS, "seed": SEED, "block_hours": BLOCK,
            "winner": winner, "selected": choice, "eligible": eligible, "metrics": scores, "a_b_fits": len(times),
            "final_fits": int(choice is not None), "fit_times": times, "input_sha256": input_hashes,
            "validation_observed_hours": stress_counts, "model_sha256": checkpoint,
        })
        stage.rename(FITTED)
    print(f"P34 selection: {winner}; {len(eligible)} eligible; {scores[winner]}", flush=True)


if __name__ == "__main__":
    selection()
