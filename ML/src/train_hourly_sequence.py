"""Train and select a small 48-hour convolutional model using CPU PyTorch."""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "p33_hourly_sequence"
CHECKPOINTS = (8, 16, 24)
MIXTURES = (0.15, 0.25, 0.50, 1.0)
POLICIES = ("plain", "augmented")
BATCH = 128
THREADS = 3
SEED = 2026111733


def digest(path):
    with path.open("rb") as stream:
        hashed = hashlib.sha256()
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(part)
    return hashed.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def moments(values):
    median = np.nanmedian(values, axis=0)
    limits = np.nanquantile(values, (0.01, 0.99), axis=0)
    spread = np.nanstd(np.clip(values, limits[0], limits[1]), axis=0)
    scale = np.where(np.isfinite(spread) & (spread > 1e-6), spread, 1.0)
    return np.nan_to_num(median).astype(np.float32), scale.astype(np.float32)


class HourlyCNN(nn.Module):
    def __init__(self, d_center, d_scale, s_center, s_scale):
        super().__init__()
        self.register_buffer("d_center", torch.from_numpy(d_center[None, :, None]))
        self.register_buffer("d_scale", torch.from_numpy(d_scale[None, :, None]))
        self.register_buffer("s_center", torch.from_numpy(s_center[None, :]))
        self.register_buffer("s_scale", torch.from_numpy(s_scale[None, :]))
        self.conv = nn.Sequential(
            nn.Conv1d(2 * len(d_center), 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(24, 24, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(192 + 2 * len(s_center), 48), nn.GELU(), nn.Dropout(0.2), nn.Linear(48, 1))

    def forward(self, dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        present = torch.isfinite(dynamic)
        scaled = torch.where(present, ((dynamic - self.d_center) / self.d_scale).clamp(-8, 8), 0.0)
        temporal = self.conv(torch.cat((scaled, present.to(scaled.dtype)), dim=1))
        present_static = torch.isfinite(static)
        scaled_static = torch.where(present_static, ((static - self.s_center) / self.s_scale).clamp(-8, 8), 0.0)
        average = F.avg_pool1d(temporal, 12).flatten(1)
        maximum = F.max_pool1d(temporal, 12).flatten(1)
        flat = torch.cat((average, maximum, scaled_static, present_static.to(scaled_static.dtype)), dim=1)
        return self.head(flat).squeeze(-1)


def masked(cube, hidden):
    return np.where(hidden[:, :, None], np.nan, cube)


def predict(model, dynamic, static):
    model.eval()
    with torch.inference_mode():
        tensors = zip(dynamic.split(BATCH), static.split(BATCH), strict=True)
        return torch.cat([torch.sigmoid(model(block, base)) for block, base in tensors]).numpy()


def fit(data, train, fold, policy, epochs, views, *, alternate_cube=None, seed=SEED):
    torch.manual_seed(seed + fold)
    cube, static = data["cube"][train], data["static"][train]
    observations = cube.transpose(0, 2, 1).reshape(-1, cube.shape[1])
    model = HourlyCNN(*moments(observations), *moments(static)).cpu()
    label = data["y"][train].astype(np.float32)
    if policy == "augmented":
        modified = masked(cube, data["aug_hidden"][fold, train]) if alternate_cube is None else alternate_cube
        cube = np.concatenate((cube, modified))
        static = np.concatenate((static, data["aug_static"][fold, train]))
        label = np.tile(label, 2)
        weight = np.repeat((0.75, 0.25), len(label) // 2).astype(np.float32)
    else:
        weight = np.ones(len(label), dtype=np.float32)
    dynamic, static, label, weight = [torch.from_numpy(item) for item in (cube, static, label, weight)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.005)
    predictions = {}
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(label)).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            losses = F.binary_cross_entropy_with_logits(model(dynamic[batch], static[batch]), label[batch], reduction="none")
            loss = (losses * weight[batch]).sum() / weight[batch].sum()
            loss.backward()
            optimizer.step()
        if epoch in CHECKPOINTS and views:
            predictions[epoch] = np.stack([predict(model, dyn, st) for dyn, st in views])
    return model.eval(), predictions


def score(y, folds, views, probabilities, weights):
    result = {}
    for view, scores in zip(views, probabilities, strict=True):
        result[view] = float(np.mean([average_precision_score(y[folds == fold], scores[folds == fold]) for fold in (0, 1, 2)]))
    result["shift"] = (result.pop("shift1") + result.pop("shift2")) / 2
    result["mapped"] = (result.pop("mapped1") + result.pop("mapped2")) / 2
    result["objective"] = sum(amount * result[key] for key, amount in weights.items())
    return result


def validate_inputs(data, manifest):
    n = manifest["stays"]
    if (
        data["cube"].shape != (n, 20, 48) or data["y"].shape != (n,)
        or set(data["folds"]) != {0, 1, 2} or set(data["y"]) != {0, 1}
        or int(data["y"].sum()) != manifest["deaths"] or manifest["dataset"] != "P12 A/B"
    ):
        raise ValueError("Expected the fixed A/B input bundle")
    actual = score(data["y"], data["folds"], manifest["views"], data["anchor"], manifest["weights"])
    for key, expected in manifest["anchor"].items():
        np.testing.assert_allclose(actual[key], expected, rtol=0, atol=1e-12)


def train():
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    inputs, output = ROOT / "inputs", ROOT / "fitted"
    if output.exists():
        raise FileExistsError(output)
    bundle_path, manifest_path = inputs / "ab.npz", inputs / "manifest.json"
    manifest = read_json(manifest_path)
    if digest(bundle_path) != manifest["bundle_sha256"]:
        raise ValueError("A/B sequence bundle was changed")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        data = dict(bundle)
    validate_inputs(data, manifest)
    y, folds = data["y"], data["folds"]
    candidates = [(policy, epoch) for policy in POLICIES for epoch in CHECKPOINTS]
    oof = {(policy, epoch): np.full(data["anchor"].shape, np.nan) for policy, epoch in candidates}
    times = []
    for fold in range(manifest["n_folds"]):
        test = folds == fold
        views = [
            (torch.from_numpy(masked(data["cube"][test], hidden[test])), torch.from_numpy(static[test]))
            for hidden, static in zip(data["val_hidden"], data["val_static"], strict=True)
        ]
        for policy in POLICIES:
            start = time.perf_counter()
            _, predicted = fit(data, ~test, fold, policy, CHECKPOINTS[-1], views)
            for epoch, values in predicted.items():
                oof[policy, epoch][:, test] = values
            seconds = time.perf_counter() - start
            times.append({"fold": fold, "policy": policy, "seconds": seconds})
            print(f"A/B fold {fold} {policy}: {seconds:.1f}s", flush=True)
    if not all(np.isfinite(values).all() for values in oof.values()):
        raise ValueError("Incomplete neural out-of-fold predictions")

    weights = manifest["weights"]
    anchor = data["anchor"]
    options = {
        f"{policy}:{epoch}:{mix:.2f}": (policy, epoch, mix)
        for policy, epoch in candidates for mix in MIXTURES
    }
    scores = {"p28": score(y, folds, manifest["views"], anchor, weights)}
    for name, (policy, epoch, mix) in options.items():
        combined = (1.0 - mix) * anchor + mix * oof[policy, epoch]
        scores[name] = score(y, folds, manifest["views"], combined, weights)
    eligible = [
        tag for tag, metrics in scores.items()
        if all(metrics[key] >= scores["p28"][key] - gap for key, gap in manifest["tolerance"].items())
    ]
    winner = max(eligible, key=lambda name: (scores[name]["objective"], scores[name]["clean"]))
    choice = options.get(winner)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="sequence-fit-") as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        model_sha = None
        if choice:
            policy, epoch, _ = choice
            model, _ = fit(data, np.ones(len(y), dtype=bool), manifest["n_folds"], policy, epoch, ())
            frozen = torch.jit.script(model)
            path = stage / "sequence.pt"
            frozen.save(str(path))
            loaded = torch.jit.load(str(path), map_location="cpu")
            mini_dynamic, mini_static = torch.from_numpy(data["cube"][:32]), torch.from_numpy(data["static"][:32])
            np.testing.assert_array_equal(predict(loaded, mini_dynamic, mini_static), predict(model, mini_dynamic, mini_static))
            model_sha = digest(path)
        if digest(bundle_path) != manifest["bundle_sha256"]:
            raise ValueError("A/B sequence bundle changed")
        save_json(stage / "manifest.json", {
            "device": "cpu", "torch": torch.__version__, "dataset": "P12 A/B", "threads": THREADS, "seed": SEED,
            "winner": winner, "selected": choice, "metrics": scores, "eligible": eligible, "fit_times": times,
            "input_manifest_sha256": digest(manifest_path), "model_sha256": model_sha, "final_fits": int(choice is not None),
        })
        stage.rename(output)
    print(f"Sequence A/B selection: {winner}; {len(eligible)} eligible; {scores[winner]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    train()
