"""Replay five frozen KEDGN seeds, then infer Set C and aligned eICU on CPU."""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ML = Path(__file__).resolve().parents[1]
AUTHOR = ML / "artifacts/related_work/KEDGN/cpu_paper_01/code"
OUTPUT = ML / "artifacts/related_work/KEDGN/frozen_transfer_20261121"
sys.path.insert(0, str(AUTHOR))
from model import KEDGN
from utils import getStats, getStats_static, get_data_split, tensorize_normalize_extract_feature, evaluate_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def padded(records):
    size = max(215, max(row["length"] for row in records))
    copies = []
    for source in records:
        row = dict(source)
        row["arr"] = np.zeros((size, 36), dtype=np.float64)
        row["time"] = np.zeros((size, 1), dtype=np.float64)
        row["arr"][:source["length"]] = source["arr"][:source["length"]]
        row["time"][:source["length"]] = source["time"][:source["length"]]
        copies.append(row)
    return copies


def probabilities(model, inputs, embeddings, batch_size):
    with torch.no_grad():
        logits = evaluate_model(model, *inputs[:5], embeddings, batch_size=batch_size).numpy().astype(np.float64)
    # Preserve the author's final two-class softmax while avoiding overflow.
    logits -= logits.max(axis=1, keepdims=True)
    unscaled = np.exp(logits)
    return (unscaled / unscaled.sum(axis=1, keepdims=True))[:, 1]


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((OUTPUT / "inputs.json").read_text(encoding="utf-8"))
    for filename, digest in manifest["files_sha256"].items():
        if sha(OUTPUT / filename) != digest:
            raise ValueError("Prepared KEDGN input changed")
    base = AUTHOR / "data/physionet"
    train, _, test, ytrain, _, ytest = get_data_split(str(base), "/splits/phy12_split5.npy", dataset="physionet")
    training_ids = set(int(row["id"]) for row in train)
    dynamic = np.stack([row["arr"] for row in train])
    static = np.asarray([row["extended_static"] for row in train])
    mf, stdf = getStats(dynamic)
    ms, ss = getStats_static(static, dataset="physionet")
    stats = mf, stdf, ms, ss
    if not all(np.isfinite(value).all() for value in stats):
        raise ValueError("Frozen training statistics contain nonfinite values")
    np.savez_compressed(OUTPUT / "training_statistics.npz", mf=mf, stdf=stdf, ms=ms, ss=ss)
    del dynamic, static, train
    embeddings = torch.load(base / "physionet_bert_var_rep_gpt_source.pt", map_location="cpu")
    source_log = AUTHOR.parent / "train.log"
    original = re.findall(r"Testing: AUROC = ([\d.]+) \| AUPRC = ([\d.]+)", source_log.read_text(encoding="utf-8"))
    if len(original) != 5:
        raise ValueError("Expected five completed author tests")
    replay_inputs = tensorize_normalize_extract_feature(copy.deepcopy(test), ytest, *stats)
    models, checkpoints, replay = [], {}, []
    for seed in range(5):
        candidates = list((AUTHOR / "models").glob(f"_physionet_*_{seed}.pt"))
        selected = max(candidates, key=lambda p: int(p.stem.split("_")[-2]))
        checkpoints[str(selected)] = sha(selected)
        model = KEDGN(DEVICE=torch.device("cpu"), hidden_dim=8, num_of_variables=36,
                      num_of_timestamps=215, d_static=9, n_class=2, rarity_alpha=1,
                      query_vector_dim=5, node_emb_dim=9, plm_rep_dim=768)
        model.load_state_dict(torch.load(selected, map_location="cpu"), strict=True)
        model.eval()
        values = probabilities(model, replay_inputs, embeddings, 256)
        metrics = [roc_auc_score(ytest, values) * 100, average_precision_score(ytest, values) * 100]
        np.testing.assert_allclose(metrics, [float(item) for item in original[seed]], rtol=0, atol=0.0051)
        replay.append({"seed": seed, "auroc_percent": metrics[0], "ap_percent": metrics[1]})
        models.append(model)
        print(f"Verified frozen seed {seed} on original 399 author test stays: {metrics}", flush=True)
    del replay_inputs, test
    results = {}
    for name in ("set_c", "eicu_hourly", "eicu_native"):
        destination = OUTPUT / f"{name}_predictions.npz"
        if destination.exists():
            with np.load(destination) as saved:
                if saved["probabilities"].shape != (5, manifest["cohorts"][name]["stays"]):
                    raise ValueError("Existing prediction shape differs")
            print(f"Retaining completed {name} inference", flush=True)
            results[name] = {"sha256": sha(destination)}
            continue
        records = np.load(OUTPUT / f"{name}_records.npy", allow_pickle=True)
        with np.load(OUTPUT / f"{name}_labels.npz") as labels:
            ids, y = labels["ids"], labels["y"]
        if name == "set_c" and set(ids) & training_ids:
            raise ValueError("Set C overlaps KEDGN training encounters")
        order = np.argsort([row["length"] for row in records], kind="stable")
        predictions = np.full((5, len(records)), np.nan)
        for start in range(0, len(order), 64):
            positions = order[start:start + 64]
            inputs = tensorize_normalize_extract_feature(padded(records[positions]), y[positions, None], *stats)
            for seed, model in enumerate(models):
                predictions[seed, positions] = probabilities(model, inputs, embeddings, 64)
            print(f"{name}: {min(start + 64, len(order))}/{len(order)} stays, all five seeds", flush=True)
        if not (np.isfinite(predictions) & (predictions >= 0) & (predictions <= 1)).all():
            raise ValueError("Invalid KEDGN probabilities")
        staged = destination.with_suffix(".pending.npz")
        np.savez_compressed(staged, ids=ids, y=y, probabilities=predictions)
        os.replace(staged, destination)
        results[name] = {"sha256": sha(destination)}
    for path, digest in checkpoints.items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen checkpoint changed during inference")
    for path, digest in manifest["source_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen inference source changed")
    report = {"device": "cpu", "model_fits": 0, "training_stays": len(training_ids),
              "checkpoints_sha256": checkpoints, "author_replay": replay,
              "source_log_sha256": sha(source_log), "predictions": results,
              "primary_prediction": "Unweighted mean of all five frozen seed probabilities"}
    (OUTPUT / "inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Frozen KEDGN transfer inference completed.", flush=True)


if __name__ == "__main__":
    main()
