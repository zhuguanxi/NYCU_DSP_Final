"""Test removal of measurement channels with different recording patterns."""

import itertools
import tempfile
from pathlib import Path

import numpy as np
import torch

from select_sequence_blend import SELECTION as P37, verify
from select_sequence_gate import PROFILE_VIEWS, PROFILES, profiles_for_fold, summaries
from sequence_blending import blend
from train_hourly_sequence import ROOT as P33, THREADS, digest, fit, predict, read_json, save_json, validate_inputs
from train_sequence_cadence import cadence_cube, tensors

ROOT = P33.parent / "p38_sequence_channel_ablation"
FITTED = ROOT / "fitted"
EXCLUDED = {"without_urine": ("Urine",), "without_rr": ("RespRate",), "without_both": ("Urine", "RespRate")}
WEIGHTS = (0.10, 0.15, 0.25, 0.40)
REFERENCES = ("p28", "p33", "p34", "p37")


def run():
    if FITTED.exists():
        raise FileExistsError(FITTED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    previous = read_json(P37)
    hashes = {**previous["input_sha256"], P37.relative_to(ROOT.parent).as_posix(): digest(P37)}
    verify(hashes)
    with np.load(P33 / "inputs/ab.npz", allow_pickle=False) as bundle:
        data = dict(bundle)
    manifest = read_json(P33 / "inputs/manifest.json")
    validate_inputs(data, manifest)
    profile = read_json(PROFILES)
    y, folds = data["y"], data["folds"]
    views = (*manifest["views"], *PROFILE_VIEWS)
    groups = manifest["groups"]
    if set(groups).intersection(EXCLUDED["without_both"]) != set(EXCLUDED["without_both"]):
        raise ValueError("Expected urine and respiratory rate in the original channel order")
    keep = {name: [i for i, group in enumerate(groups) if group not in excluded] for name, excluded in EXCLUDED.items()}
    profile_anchors = [manifest["views"].index(name) for name in ("mapped1", "mapped2")]
    anchor = np.concatenate((data["anchor"], data["anchor"][profile_anchors]))
    controls = {tag: previous["scores"][tag if tag != "p37" else previous["winner"]] for tag in REFERENCES}
    for name, value in summaries(y, folds, views, anchor).items():
        np.testing.assert_allclose(value, controls["p28"][name], rtol=0, atol=1e-12)
    coarse = cadence_cube(data["cube"])
    prepared = {name: {**data, "cube": coarse[:, indices]} for name, indices in keep.items()}
    forecasts = {name: np.full(anchor.shape, np.nan) for name in keep}
    fits = []
    for fold in range(manifest["n_folds"]):
        test = folds == fold
        validation = profiles_for_fold(data, manifest, test, profile, fold)
        if tuple(validation) != views:
            raise ValueError("A/B validation views changed")
        coarse_views = {name: (cadence_cube(cube), static) for name, (cube, static) in validation.items()}
        for tag, indices in keep.items():
            projected = {name: (cube[:, indices], static) for name, (cube, static) in coarse_views.items()}
            _, predicted = fit(prepared[tag], ~test, fold, "plain", 8, tensors(projected, False))
            forecasts[tag][:, test] = predicted[8]
            fits.append({"fold": fold, "ablation": tag})
            print(f"P38 fitted {tag}, A/B fold {fold}", flush=True)
    if not all((np.isfinite(p) & (p >= 0) & (p <= 1)).all() for p in forecasts.values()):
        raise ValueError("Invalid or incomplete ablation fold predictions")

    scores = {**controls}
    choices = {}
    for tag, weight, mode in itertools.product(EXCLUDED, WEIGHTS, ("probability", "log_odds")):
        key = f"{tag}:{weight:.2f}:{mode}"
        rule = {"ablation": tag, "mode": mode, "weight": weight, "bound": 2.0 if mode == "log_odds" else None}
        choices[key] = rule
        scores[key] = summaries(y, folds, views, blend(anchor, forecasts[tag], rule))
    eligible = [
        tag for tag in ("p33", "p34", "p37", *choices)
        if all(scores[tag][name] >= scores["p33"][name] - tol for name, tol in manifest["tolerance"].items())
    ]
    winner = max(eligible, key=lambda tag: (scores[tag]["objective"], scores[tag]["clean"]))
    selected = choices.get(winner)
    model_hash, included = None, None
    ROOT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="channel-ablation-") as temporary:
        staged = Path(temporary) / FITTED.name
        staged.mkdir()
        if selected is not None:
            tag = selected["ablation"]
            final_data = prepared[tag]
            model, _ = fit(final_data, np.ones(len(y), dtype=bool), manifest["n_folds"], "plain", 8, ())
            path = staged / "sequence.pt"
            torch.jit.script(model).save(str(path))
            restored = torch.jit.load(str(path), map_location="cpu")
            sample = torch.from_numpy(final_data["cube"][:16]), torch.from_numpy(final_data["static"][:16])
            np.testing.assert_array_equal(predict(model, *sample), predict(restored, *sample))
            model_hash = digest(path)
            included = [groups[i] for i in keep[tag]]
        verify(hashes)
        save_json(staged / "manifest.json", {
            "dataset": "P12 A/B", "device": "cpu", "threads": THREADS, "winner": winner,
            "selected": selected, "eligible": eligible, "candidates": choices, "scores": scores,
            "excluded_by_candidate": EXCLUDED, "included_channels": included, "block_hours": 4,
            "a_b_fits": fits, "final_fits": int(selected is not None), "model_sha256": model_hash,
            "tolerance": manifest["tolerance"], "objective_weights": previous["objective_weights"], "input_sha256": hashes,
        })
        staged.rename(FITTED)
    print(f"P38 A/B selected {winner}, {len(eligible)} eligible: {scores[winner]}", flush=True)


if __name__ == "__main__":
    run()
