"""Prepare unscaled, aligned 48-hour inputs for the common mortality tests."""

import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "src"))

from eicu_demo_features import DEMO, MODES, STAY, build_features
from evaluate_eicu_demo import source_hashes
from evaluate_eicu_urine_mapping import urine_events
from prepare import DATA, OUT, extract_record, sha

OUTPUT = OUT / "related_work/common_sequence_inputs"
EVENTS = ["RecordID", "elapsed_minute", "variable", "clean_value"]
STATIC = ["static_Age", "static_Gender", "static_Height", "static_ICUType", "static_Weight"]
SOURCE = OUT / "related_work/paper_data/P12data/processed_data/ts_params.npy"


def filtered(frame, channels, index):
    valid = frame["variable"].isin(channels) & frame["elapsed_minute"].between(0, 2880)
    if "quality_flag" in frame:
        valid &= frame["quality_flag"].eq("valid")
    result = frame.loc[valid, EVENTS].dropna(subset=["clean_value"]).copy()
    result["RecordID"] = result["RecordID"].astype("int64")
    result["elapsed_minute"] = result["elapsed_minute"].astype("int32")
    result["clean_value"] = result["clean_value"].astype("float64")
    if not result.RecordID.isin(index).all() or not np.isfinite(result.clean_value).all():
        raise ValueError("Invalid sequence event key or value")
    return result


def write_events(path, frames):
    count = 0
    writer = None
    try:
        for frame in frames:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if not count:
        raise ValueError("Prepared an empty cohort's sequence inputs")
    return count


def p12_ab(channels, index):
    source = pq.ParquetFile(OUT / "observations_AB.parquet")
    for batch in source.iter_batches(batch_size=150000, columns=[*EVENTS, "quality_flag"]):
        yield filtered(batch.to_pandas(), channels, index)


def p12_c(channels, index):
    rows = []
    seen = set()
    with tarfile.open(DATA / "set-c.tar.gz", "r:gz") as archive:
        for member in archive:
            rid = Path(member.name).stem
            if not member.isfile() or not rid.isdigit():
                continue
            record_id = int(rid)
            if record_id in seen:
                raise ValueError("P12 C repeats an encounter")
            seen.add(record_id)
            source = archive.extractfile(member)
            _, records, _ = extract_record(source.read(), record_id)
            rows.extend((stay, minute, channel, value) for stay, minute, channel, _, value, flag, _ in records
                        if flag == "valid" and channel in channels)
            if len(seen) % 200 == 0:
                yield filtered(pd.DataFrame(rows, columns=EVENTS), channels, index)
                rows = []
    if rows:
        yield filtered(pd.DataFrame(rows, columns=EVENTS), channels, index)
    if seen != set(index):
        raise ValueError("P12 C sequence cohort differs from the frozen comparison")


def save_cohort(folder, name, features, labels):
    labels = labels.loc[features.index]
    if not features.index.is_unique or set(labels.unique()) != {0, 1}:
        raise ValueError("Invalid cohort labels")
    features.loc[:, STATIC].to_parquet(folder / f"{name}_static.parquet")
    labels.rename("y").to_frame().to_parquet(folder / f"{name}_labels.parquet")
    return {"stays": len(labels), "deaths": int(labels.sum())}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    channels = np.load(SOURCE, allow_pickle=True).tolist()
    aliases = {"TropI": "TroponinI", "TropT": "TroponinT"}
    channels = [aliases.get(name, name) for name in channels]
    if len(channels) != 36 or len(set(channels)) != 36 or "Weight" in channels:
        raise ValueError("Unexpected author dynamic-channel schema")
    sources = [
        SOURCE, OUT / "observations_AB.parquet", OUT / "features_AB.parquet",
        OUT / "labels_AB.csv", OUT / "folds3.csv", OUT / "C_evaluation/features_C.parquet",
        OUT / "C_evaluation/labels_C.csv", DATA / "set-c.tar.gz",
    ]
    digests = {str(path): sha(path) for path in sources}
    demo_digests = source_hashes(DEMO)
    reference = json.loads((OUT / "eicu_demo_external/summary.json").read_text(encoding="utf-8"))
    if demo_digests != reference["data_sha256"]:
        raise ValueError("eICU release differs from the original external evaluation")
    ab = pd.read_parquet(OUT / "features_AB.parquet")
    c = pd.read_parquet(OUT / "C_evaluation/features_C.parquet")
    if set(ab.index) & set(c.index):
        raise ValueError("The common development and test cohorts overlap")
    lab = pd.read_csv(OUT / "labels_AB.csv", index_col="RecordID").In_hospital_death
    lc = pd.read_csv(OUT / "C_evaluation/labels_C.csv", index_col="RecordID").In_hospital_death
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent, prefix="common-inputs-") as temporary:
        stage = Path(temporary) / OUTPUT.name
        stage.mkdir()
        cohorts = {
            "p12_ab": save_cohort(stage, "p12_ab", ab, lab),
            "p12_c": save_cohort(stage, "p12_c", c, lc),
        }
        assert cohorts["p12_ab"] == {"stays": 8000, "deaths": 1122}
        assert cohorts["p12_c"] == {"stays": 4000, "deaths": 585}
        for name, frames in (("p12_ab", p12_ab(channels, ab.index)), ("p12_c", p12_c(channels, c.index))):
            cohorts[name]["events"] = write_events(stage / f"{name}_events.parquet", frames)
            print(f"Prepared {name}: {cohorts[name]}", flush=True)
        folds = pd.read_csv(OUT / "folds3.csv", index_col="RecordID").loc[ab.index]
        if not set(folds.fold3.unique()) == {0, 1, 2}:
            raise ValueError("Expected the original P12 A/B three folds")
        folds.to_parquet(stage / "p12_ab_folds.parquet")
        captured = []
        def collect(events, index):
            frame = events.rename(columns={STAY: "RecordID", "minute": "elapsed_minute", "value": "clean_value"})
            captured.append(filtered(frame, channels, index))
            return pd.DataFrame(index=index)
        cohort, features, _, _ = build_features(DEMO, collect)
        assert len(captured) == len(MODES)
        if len(cohort) != 782 or int(cohort.y.sum()) != 91:
            raise ValueError("eICU common test cohort differs from the original")
        urine, _ = urine_events(cohort.index)
        urine = filtered(urine.rename(columns={STAY: "RecordID", "minute": "elapsed_minute", "value": "clean_value"}),
                         channels, cohort.index)
        for mode, events in zip(MODES, captured):
            name = f"eicu_{mode}"
            cohorts[name] = save_cohort(stage, name, features[mode], cohort.y)
            cohorts[name]["events"] = write_events(stage / f"{name}_events.parquet", [events, urine])
        cohort[["uniquepid", "hospitalid"]].to_parquet(stage / "eicu_groups.parquet")
        if any(sha(path) != digests[str(path)] for path in sources) or source_hashes(DEMO) != demo_digests:
            raise ValueError("Input sources changed during preparation")
        manifest = {
            "device": "cpu", "end_minute": 2880, "dynamic_channels": channels, "static_columns": STATIC,
            "cohorts": cohorts, "source_sha256": digests, "eicu_sha256": demo_digests,
            "normalization_fitted": False, "model_fits": 0, "test_predictions": False,
            "files_sha256": {path.name: sha(path) for path in stage.iterdir()},
            "next_stage": "Model-specific tensorization, A/B-only fitting, then held-out evaluation",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        stage.rename(OUTPUT)
    print(f"Saved unscaled common sequence inputs to {OUTPUT}", flush=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
