"""Reuse the exact fitted P12 transformations in the upstream MA-GBT runtime."""

import json
import os
import tempfile
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np

from prepare import OUT, sha
from train_missingness_augmentation import (
    AB_FEATURES,
    AB_FOLDS,
    AB_LABELS,
    AB_SELECTION,
    FOLD_IDS,
    FROZEN,
    MODELS,
    ab_inputs,
    from_json,
    verified_template,
)

RUN = OUT / "ma_gbt_p12"
SCHEMA_FILE = RUN / "encoded_schemas.json"


def digests(paths):
    return {path.relative_to(OUT).as_posix(): sha(path) for path in paths}


def schema_for(pre, columns):
    by_name = {name: (step, list(cols)) for name, step, cols in pre.transformers_}
    num, numeric = by_name["num"]
    cat, categorical = by_name["cat"]
    if list(num.named_steps) != ["impute"] or list(cat.named_steps) != ["impute", "encode"]:
        raise ValueError("Unexpected fitted P12 preprocessing steps")
    enc = cat.named_steps["encode"]
    if enc.handle_unknown != "ignore" or enc.drop != "first" or enc.sparse_output:
        raise ValueError("Unexpected fitted P12 one-hot options")
    return {
        "columns": list(columns),
        "numeric": numeric,
        "medians": num.named_steps["impute"].statistics_.tolist(),
        "categorical": [
            {"name": name, "fill": float(fill), "levels": np.delete(levels, drop).tolist()}
            for name, fill, levels, drop in zip(
                categorical, cat.named_steps["impute"].statistics_,
                enc.categories_, enc.drop_idx_, strict=True
            )
        ],
    }


def encode(frame, schema, dtype=np.float32):
    numeric = frame.loc[:, schema["numeric"]].to_numpy(dtype=np.float64)
    numeric_mask = np.isnan(numeric)
    values = [np.where(numeric_mask, schema["medians"], numeric)]
    masks = [numeric_mask]
    for cat in schema["categorical"]:
        raw = frame[cat["name"]].to_numpy(dtype=np.float64)
        missing = np.isnan(raw)[:, None]
        filled = np.where(missing, cat["fill"], raw[:, None])
        dummy = filled == np.asarray(cat["levels"])[None, :]
        values.append(dummy)
        masks.append(np.broadcast_to(missing, dummy.shape))
    encoded = np.ascontiguousarray(np.column_stack(values), dtype=dtype)
    mask = np.ascontiguousarray(np.column_stack(masks), dtype=np.int8)
    if not np.isfinite(encoded).all():
        raise ValueError("The fitted P12 transformation produced nonfinite input")
    return encoded, mask


def verify_encoding(frame, pre, schema):
    subset = frame.loc[:, schema["columns"]]
    probe = subset.iloc[:2].copy()
    for cat in schema["categorical"]:
        column = cat["name"]
        probe[column] = [np.nan, 999.0]
    for check in (subset, probe):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*")
            expected = pre.transform(check)
        actual, mask = encode(check, schema, dtype=np.float64)
        np.testing.assert_array_equal(actual, expected)
        count = len(schema["numeric"])
        np.testing.assert_array_equal(mask[:, :count], check[schema["numeric"]].isna())
        for cat in schema["categorical"]:
            width = len(cat["levels"])
            np.testing.assert_array_equal(
                mask[:, count:count + width],
                np.repeat(check[cat["name"]].isna().to_numpy()[:, None], width, axis=1),
            )
            count += width


def export_schemas():
    if RUN.exists():
        raise FileExistsError(RUN)
    frame, _, _ = ab_inputs()
    feature_hash = sha(AB_FEATURES)
    selection = from_json(AB_SELECTION)
    sources = [AB_FEATURES, AB_LABELS, AB_FOLDS, AB_SELECTION]
    schemas = {}
    for name in MODELS:
        folder = OUT / "runs" / selection[f"XGBoost_{name}"]
        config = folder / "config.json"
        expected = {
            "model": "XGBoost", "family": 2, "strategy": name,
            "nfold": 3, "adult": False, "data_sha": feature_hash,
        }
        actual = from_json(config)
        if any(actual[key] != value for key, value in expected.items()):
            raise ValueError(f"{name} fold source does not match the original A/B experiment")
        sources.append(config)
        schemas[name] = {}
        paths = {str(fold): folder / f"fold{fold}.joblib" for fold in FOLD_IDS}
        paths["final"] = FROZEN / f"{name}.joblib"
        for fold, path in paths.items():
            pipe, columns = verified_template(path, feature_hash)
            pre = pipe.named_steps["pre"]
            schema = schema_for(pre, columns)
            verify_encoding(frame, pre, schema)
            sources.append(path)
            schemas[name][fold] = schema
            print(f"{name} {fold}: all {len(frame)} A/B rows exactly match the frozen preprocessing", flush=True)
    manifest = {"rows_checked_per_schema": len(frame), "source_sha256": digests(sources), "schemas": schemas}
    with tempfile.TemporaryDirectory(dir=OUT, prefix="ma-gbt-schemas-") as temporary:
        from pathlib import Path

        staging = Path(temporary) / "output"
        staging.mkdir()
        (staging / SCHEMA_FILE.name).write_text(
            json.dumps(manifest, allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(RUN)


if __name__ == "__main__":
    export_schemas()
