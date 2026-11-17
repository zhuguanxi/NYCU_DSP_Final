"""Fit one fixed CPU XGBoost baseline on each published mortality split."""

import json
import pickle
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from run_cpu import ML, PAPER, ROOT

ARTIFACTS = ML / "artifacts"
OUTPUT = ROOT / "matched_xgboost"
LABEL = "In_hospital_death"


def array_splits(dataset, names, deaths):
    labels = np.load(dataset / "processed_data" / "arr_outcomes.npy", allow_pickle=True)
    ids = labels[:, 0].astype(int)
    np.testing.assert_array_equal(deaths.loc[ids], labels[:, -1].astype(int))
    if len(np.unique(ids)) != len(ids):
        raise ValueError("An author cohort repeats record IDs")
    return {
        name: tuple(ids[np.asarray(indices, dtype=int)] for indices in np.load(
            dataset / "splits" / f"{split}.npy", allow_pickle=True,
        ))
        for name, split in names.items()
    }


def author_splits(deaths):
    splits = array_splits(
        PAPER / "P12data", {f"raindrop_{i}": f"phy12_split{i}" for i in range(1, 6)}, deaths,
    )
    splits.update(array_splits(
        ROOT / "KEDGN/upstream/data/physionet", {"kedgn_physionet_5": "phy12_split5"}, deaths,
    ))
    warp = ROOT / "Warpformer/upstream/data_utils"
    train_and_val, test = (
        np.asarray(pickle.loads((warp / f"{name}_record_id.pkl").read_bytes()), dtype=int)
        for name in ("train", "test")
    )
    boundary = int(0.8 * len(train_and_val))
    splits["warpformer_set_a"] = (train_and_val[:boundary], train_and_val[boundary:], test)
    return splits


def data():
    x = pd.concat([
        pd.read_parquet(ARTIFACTS / "features_AB.parquet"),
        pd.read_parquet(ARTIFACTS / "C_evaluation/features_C.parquet"),
    ], verify_integrity=True)
    y = pd.concat([
        pd.read_csv(ARTIFACTS / "labels_AB.csv", index_col="RecordID")[LABEL],
        pd.read_csv(ARTIFACTS / "C_evaluation/labels_C.csv", index_col="RecordID")[LABEL],
    ], verify_integrity=True)
    return x, y.loc[x.index]


def estimate(x, y, template, train, validation, test):
    ids = np.concatenate((train, validation, test))
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Author's training, validation, and test samples are not disjoint")
    fitted = make_pipeline(
        clone(template),
        XGBClassifier(device="cpu", tree_method="hist", n_jobs=2, random_state=42, eval_metric="logloss"),
    )
    fitted.fit(x.loc[train], y.loc[train])
    probs = fitted.predict_proba(x.loc[test])[:, 1]
    target = y.loc[test].to_numpy()
    summary = {
        "stays": [len(part) for part in (train, validation, test)], "test_deaths": int(target.sum()),
        "ap": float(average_precision_score(target, probs)), "auroc": float(roc_auc_score(target, probs)),
    }
    return summary, probs


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    x, y = data()
    original, _, columns = joblib.load(ARTIFACTS / "final_models/Full.joblib")
    if len(columns) != 435 or set(columns) != set(x.columns):
        raise ValueError("Expected the original Full feature schema")
    x = x.loc[:, columns]
    template = original["pre"]
    results = {}
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="matched-") as temp:
        stage = Path(temp) / OUTPUT.name
        stage.mkdir()
        for name, parts in author_splits(y).items():
            summary, probs = estimate(x, y, template, *parts)
            results[name] = summary
            np.savez_compressed(stage / f"{name}.npz", ids=parts[-1], probability=probs)
            print(name, summary, flush=True)
        raindrop = np.asarray([[row["ap"], row["auroc"]] for key, row in results.items() if key.startswith("raindrop_")])
        results["raindrop_average"] = {
            "mean_ap": raindrop[:, 0].mean(), "sd_ap": raindrop[:, 0].std(),
            "mean_auroc": raindrop[:, 1].mean(), "sd_auroc": raindrop[:, 1].std(),
        }
        (stage / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        stage.rename(OUTPUT)


if __name__ == "__main__":
    main()
