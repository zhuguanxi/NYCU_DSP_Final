"""Prepare author-compatible records for frozen KEDGN transfer without fitting."""

import io
import json
import argparse
from pathlib import Path
import sys
import tarfile
import tempfile

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "src"))
from prepare import DATA, OUT, sha

AUTHOR = OUT / "related_work/KEDGN/cpu_paper_01/code"
COMMON = OUT / "related_work/common_sequence_inputs"
OUTPUT = OUT / "related_work/KEDGN/frozen_transfer_20261121"
STATIC = ("Age", "Gender", "Height", "ICUType", "Weight")
EVENTS = ["elapsed_minute", "variable", "clean_value"]


def extended(values):
    age, gender, height, icu, weight = values
    return [age, int(gender == 0), int(gender == 1), height,
            *[int(icu == code) for code in (1, 2, 3, 4)], weight]


def pack(stay, admission, rows, variables):
    lookup = {name: i for i, name in enumerate(variables)}
    times = list(dict.fromkeys(minute for minute, channel, value in rows if minute < 2880 and channel in lookup))
    time_index = {minute: i for i, minute in enumerate(times)}
    size = max(215, len(times))
    array = np.zeros((size, len(variables)), dtype=np.float64)
    clock = np.zeros((size, 1), dtype=np.float64)
    for minute, channel, value in rows:
        if minute < 2880 and channel in lookup:
            i = time_index[minute]
            array[i, lookup[channel]] = value
            clock[i, 0] = minute
    return {"id": str(stay), "static": tuple(admission), "extended_static": extended(admission),
            "arr": array, "time": clock, "length": len(times)}


def raw_record(content, stay, variables):
    data = pd.read_csv(io.BytesIO(content), header=1, names=["time", "param", "value"])
    repaired_order = tuple(data.iloc[:5].param) != STATIC
    admission = {}
    for clock, variable, value in data.itertuples(index=False, name=None):
        if variable in STATIC and str(clock) == "00:00":
            admission.setdefault(variable, float(value))
    admission = [admission.get(name, -1.0) for name in STATIC]
    rows = []
    for clock, variable, value in data.itertuples(index=False, name=None):
        if variable in variables:
            hour, minute = map(float, str(clock).split(":"))
            rows.append((60 * hour + minute, variable, float(value)))
    record = pack(stay, admission, rows, variables)
    record["static_order_repaired"] = repaired_order
    return record


def write_records(stage, name, records, labels):
    if not labels.index.is_unique or len(records) != len(labels):
        raise ValueError("Cohort row alignment differs")
    np.testing.assert_array_equal([int(row["id"]) for row in records], labels.index)
    np.save(stage / f"{name}_records.npy", np.asarray(records, dtype=object), allow_pickle=True)
    np.savez_compressed(stage / f"{name}_labels.npz", ids=labels.index.to_numpy(), y=labels.to_numpy(dtype=int))
    return {
        "stays": len(labels), "deaths": int(labels.sum()),
        "zero_length_stays": sum(row["length"] == 0 for row in records),
        "maximum_length": max(row["length"] for row in records),
        "stays_over_215_times": sum(row["length"] > 215 for row in records),
        "static_order_repaired": sum(row.get("static_order_repaired", False) for row in records),
    }


def portable_records(folder):
    manifest_path = folder / "inputs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in manifest["cohorts"]:
        destination = folder / f"{name}_portable.npz"
        if destination.exists():
            raise FileExistsError(destination)
        records = np.load(folder / f"{name}_records.npy", allow_pickle=True)
        sizes = np.asarray([len(row["arr"]) for row in records], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(sizes)]
        np.savez_compressed(
            destination, offsets=offsets,
            arr=np.concatenate([row["arr"] for row in records]),
            time=np.concatenate([row["time"] for row in records]),
            static=np.asarray([row["extended_static"] for row in records], dtype=np.float64),
            lengths=np.asarray([row["length"] for row in records], dtype=np.int64),
        )
        manifest["files_sha256"][destination.name] = sha(destination)
    manifest["portable_arrays"] = True
    temporary = manifest_path.with_suffix(".pending")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    variables = np.load(AUTHOR / "data/physionet/processed_data/ts_params.npy", allow_pickle=True).tolist()
    if len(variables) != 36:
        raise ValueError("Expected 36 KEDGN variables")
    author_records = np.load(AUTHOR / "data/physionet/processed_data/PTdict_list.npy", allow_pickle=True)
    sample = author_records[np.linspace(0, len(author_records) - 1, 30, dtype=int)]
    for original in sample:
        stay = int(original["id"])
        new = raw_record((DATA / "set-a" / f"{stay}.txt").read_bytes(), stay, variables)
        for field in ("arr", "time", "static", "extended_static", "length"):
            np.testing.assert_array_equal(original[field], new[field])
    common = json.loads((COMMON / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in common["files_sha256"].items():
        if sha(COMMON / name) != digest:
            raise ValueError(f"Shared transfer input changed: {name}")
    if common["dynamic_channels"] != variables:
        raise ValueError("KEDGN and shared transfer input channel order differ")
    paths = [AUTHOR / "train.py", AUTHOR / "model.py", AUTHOR / "utils.py",
             AUTHOR / "data/physionet/processed_data/PTdict_list.npy",
             AUTHOR / "data/physionet/processed_data/arr_outcomes.npy",
             AUTHOR / "data/physionet/splits/phy12_split5.npy",
             AUTHOR / "data/physionet/physionet_bert_var_rep_gpt_source.pt",
             DATA / "set-c.tar.gz", COMMON / "manifest.json"]
    digests = {str(path): sha(path) for path in paths}
    labels = pd.read_csv(OUT / "C_evaluation/labels_C.csv", index_col="RecordID").In_hospital_death
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent, prefix="frozen-transfer-") as temporary:
        stage = Path(temporary) / OUTPUT.name
        stage.mkdir()
        rows = {}
        with tarfile.open(DATA / "set-c.tar.gz", "r:gz") as archive:
            for member in archive:
                if member.isfile() and Path(member.name).stem.isdigit():
                    stay = int(Path(member.name).stem)
                    if stay in rows:
                        raise ValueError("Duplicate Set C record")
                    rows[stay] = raw_record(archive.extractfile(member).read(), stay, variables)
        if set(rows) != set(labels.index):
            raise ValueError("Set C archive differs from frozen comparison")
        cohorts = {"set_c": write_records(stage, "set_c", [rows[stay] for stay in labels.index], labels)}
        del rows
        for mode in ("hourly", "native"):
            name = f"eicu_{mode}"
            events = pd.read_parquet(COMMON / f"{name}_events.parquet")
            static = pd.read_parquet(COMMON / f"{name}_static.parquet")
            target = pd.read_parquet(COMMON / f"{name}_labels.parquet").y
            grouped = events.groupby("RecordID", sort=False)
            records = []
            for stay in target.index:
                admission = static.loc[stay, [f"static_{key}" for key in STATIC]].fillna(-1).to_numpy(dtype=float)
                series = grouped.get_group(stay).sort_values("elapsed_minute", kind="stable")
                measurements = list(series.loc[:, EVENTS].itertuples(index=False, name=None))
                records.append(pack(stay, admission, measurements, variables))
            cohorts[name] = write_records(stage, name, records, target)
        manifest = {
            "device": "cpu", "model_fits": 0, "dynamic_channels": variables, "cohorts": cohorts,
            "source_sha256": digests, "common_input_manifest_sha256": sha(COMMON / "manifest.json"),
            "raw_parser_matches_author_sample": len(sample),
            "time_rule": "Author window minute < 2880. Keep all timestamps, including lengths > 215.",
            "missing_rule": "Author positive-value mask; static missing -1 before author transform.",
            "files_sha256": {path.name: sha(path) for path in stage.iterdir()},
        }
        if any(sha(path) != digest for path, digest in ((Path(p), h) for p, h in digests.items())):
            raise ValueError("Frozen sources changed during input preparation")
        (stage / "inputs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        stage.rename(OUTPUT)
    print(json.dumps(cohorts, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-cache", action="store_true")
    args = parser.parse_args()
    if args.portable_cache:
        portable_records(OUTPUT)
    else:
        main()
        portable_records(OUTPUT)
