"""Measure seed sensitivity and select a sequence ensemble on P12 A/B."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from select_sequence_blend import SELECTION as P37, verify
from select_sequence_gate import PROFILE_VIEWS, PROFILES, profiles_for_fold, summaries
from sequence_blending import blend
from train_hourly_sequence import ROOT as P33, SEED, THREADS, digest, fit, predict, read_json, save_json, validate_inputs
from train_sequence_cadence import cadence_cube, tensors
from train_sequence_channel_ablation import EXCLUDED, FITTED as P38

ROOT = P33.parent / "p39_sequence_seed_ensemble"
FITTED = ROOT / "fitted"
SEEDS = (SEED, SEED + 100, SEED + 200)
EXCLUSIONS = {"all_channels": (), **{key: EXCLUDED[key] for key in ("without_urine", "without_both")}}
BLEND = {"mode": "log_odds", "weight": 0.25, "bound": 2.0}


def run():
    if FITTED.exists():
        raise FileExistsError(FITTED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    path38 = P38 / "manifest.json"
    p38 = read_json(path38)
    p37 = read_json(P37)
    if (
        p37["selected"] != {"branch": "p34", **BLEND}
        or p38["selected"] != {"ablation": "without_both", **BLEND}
        or p38["objective_weights"] != p37["objective_weights"]
    ):
        raise ValueError("Expected the frozen P37 and P38 four-hour rules")
    hashes = {
        **p38["input_sha256"], path38.relative_to(ROOT.parent).as_posix(): digest(path38),
        (P38 / "sequence.pt").relative_to(ROOT.parent).as_posix(): p38["model_sha256"],
    }
    verify(hashes)
    manifest = read_json(P33 / "inputs/manifest.json")
    profile = read_json(PROFILES)
    with np.load(P33 / "inputs/ab.npz", allow_pickle=False) as bundle:
        data = dict(bundle)
    validate_inputs(data, manifest)
    y, folds = data["y"], data["folds"]
    groups, views = manifest["groups"], (*manifest["views"], *PROFILE_VIEWS)
    if not set(EXCLUSIONS["without_both"]).issubset(groups):
        raise ValueError("Expected original urine and respiratory-rate channels")
    keep = {key: [i for i, group in enumerate(groups) if group not in excluded] for key, excluded in EXCLUSIONS.items()}
    source = cadence_cube(data["cube"])
    prepared = {
        tag: {**data, "cube": source if not EXCLUSIONS[tag] else source[:, indices]}
        for tag, indices in keep.items()
    }
    profile_anchors = [manifest["views"].index(name) for name in ("mapped1", "mapped2")]
    anchor = np.concatenate((data["anchor"], data["anchor"][profile_anchors]))
    forecasts = {
        tag: np.full((len(SEEDS), *anchor.shape), np.nan, dtype=np.float64)
        for tag in EXCLUSIONS
    }
    fits = []
    for fold in range(manifest["n_folds"]):
        test = folds == fold
        validation = profiles_for_fold(data, manifest, test, profile, fold)
        if tuple(validation) != views:
            raise ValueError("Unexpected A/B validation view order")
        coarse_views = {name: (cadence_cube(cube), static) for name, (cube, static) in validation.items()}
        for tag, indices in keep.items():
            projected = {name: (cube[:, indices], static) for name, (cube, static) in coarse_views.items()}
            fold_inputs = tensors(projected if EXCLUSIONS[tag] else coarse_views, False)
            for i, seed in enumerate(SEEDS):
                _, predicted = fit(prepared[tag], ~test, fold, "plain", 8, fold_inputs, seed=seed)
                forecasts[tag][i][:, test] = predicted[8]
                fits.append({"fold": fold, "variant": tag, "seed": seed})
            print(f"P39 {tag}, A/B fold {fold}, all {len(SEEDS)} seeds fitted", flush=True)
    if not all((np.isfinite(p) & (p >= 0) & (p <= 1)).all() for p in forecasts.values()):
        raise ValueError("Invalid or incomplete seed predictions")

    scores = {tag: p38["scores"][tag] for tag in ("p28", "p33", "p34", "p37")}
    scores["p38"] = p38["scores"][p38["winner"]]
    per_seed = {
        tag: [summaries(y, folds, views, blend(anchor, item, BLEND)) for item in values]
        for tag, values in forecasts.items()
    }
    originals = {
        "all_channels": scores["p37"], "without_urine": p38["scores"]["without_urine:0.25:log_odds"],
        "without_both": scores["p38"],
    }
    for tag, reference in originals.items():
        for name, point in per_seed[tag][0].items():
            np.testing.assert_allclose(point, reference[name], rtol=0, atol=1e-12)
    for name, point in summaries(y, folds, views, anchor).items():
        np.testing.assert_allclose(point, scores["p28"][name], rtol=0, atol=1e-12)
    choices = {tag: {"variant": tag, **BLEND} for tag in EXCLUSIONS}
    for tag, values in forecasts.items():
        scores[tag] = summaries(y, folds, views, blend(anchor, values.mean(axis=0, dtype=np.float64), BLEND))
    eligible = [
        tag for tag in ("p33", "p34", "p37", "p38", *choices)
        if all(scores[tag][key] >= scores["p33"][key] - tol for key, tol in manifest["tolerance"].items())
    ]
    winner = max(eligible, key=lambda tag: (scores[tag]["objective"], scores[tag]["clean"]))
    selected = choices.get(winner)
    ROOT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="sequence-seeds-") as temporary:
        out = Path(temporary) / FITTED.name
        out.mkdir()
        models = {}
        if selected is not None:
            training = np.ones(len(y), dtype=bool)
            final_data = prepared[winner]
            sample = torch.from_numpy(final_data["cube"][:16]), torch.from_numpy(final_data["static"][:16])
            for seed in SEEDS:
                model, _ = fit(final_data, training, manifest["n_folds"], "plain", 8, (), seed=seed)
                path = out / f"sequence_{seed}.pt"
                torch.jit.script(model).save(str(path))
                restored = torch.jit.load(str(path), map_location="cpu")
                np.testing.assert_array_equal(predict(model, *sample), predict(restored, *sample))
                models[path.name] = digest(path)
        verify(hashes)
        save_json(out / "manifest.json", {
            "dataset": "P12 A/B", "device": "cpu", "block_hours": 4, "threads": THREADS,
            "seeds": SEEDS, "winner": winner, "selected": selected, "candidates": choices,
            "eligible": eligible, "scores": scores, "per_seed": per_seed, "a_b_fits": fits,
            "excluded_by_variant": EXCLUSIONS, "included_channels": [groups[i] for i in keep[winner]] if selected else None,
            "tolerance": manifest["tolerance"], "objective_weights": p38["objective_weights"],
            "model_sha256": models, "final_fits": len(models), "input_sha256": hashes,
        })
        out.rename(FITTED)
    print(f"P39 selected {winner}, {len(eligible)} eligible configurations: {scores[winner]}", flush=True)


if __name__ == "__main__":
    run()
