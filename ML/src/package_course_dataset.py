"""Export the de-identified model inputs for the course dataset delivery."""

import hashlib
import os
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_eicu_hospital_variation import hospital_blocks
from audit_p12_outcome_timing import DEATH, OFFICIAL_SHA256, cohort_masks, outcome_timing
from eicu_demo_features import DEMO, MODES
from evaluate_24h_landmark import (
    APACHE_FILE, APACHE_SHA256, ROOT as P49, demo_apache, demo_inputs_24h, development_24h, set_c_24h,
)
from evaluate_eicu_24h_augmentation import OUTPUT as P50
from evaluate_eicu_demo import source_hashes
from evaluate_hourly_sequence import eicu_sequence_cases
from evaluate_ma_gbt_p12 import REFERENCES, plan_for
from evaluate_real_eicu_augmentation import HOSPITAL_SHA256, OUTPUT as P44, partition
from evaluate_robust_ensemble import EVALUATION as P28
from evaluate_temporal_features import save_json, set_c_temporal
from pilot_temporal_features import DEST as P22, GROUPS, STATS, time_summary
from prepare import DATA, OUT, sha
from prepare_hourly_sequence import HOURS, PREPARED, STATIC_CODES, STATIC_NUM, hourly_cube, static_values
from stress_missingness_c import C_FEATURES, C_LABELS, SCENARIOS, load_cohort, mask_draws
from train_gcs_routed_temporal import VIEWS
from train_predictive_boosting import AB_FEATURES, AB_FOLDS, AB_LABELS, ab_inputs, from_json, verify_sources

OUTPUT = OUT / "course_submission" / "icu_mortality_dataset"
ARCHIVE_NAME = "icu_mortality_preprocessed_dataset.zip"
README = Path(__file__).resolve().parents[1] / "reports" / "課程資料包_README.txt"
SCORE_NAMES = ["SAPS-I", "SOFA"]
ARTIFACT_FILES = (
    AB_FEATURES, AB_FOLDS, AB_LABELS, C_FEATURES, C_LABELS, P22 / "features_AB_temporal.parquet",
    PREPARED / "ab.npz", PREPARED / "manifest.json", P28 / "summary.json",
    P44 / "summary.json", P49 / "manifest.json", P49 / "evaluation/summary.json", P50 / "summary.json",
    OUT / "feature_dictionary.csv",
)


def keys(index, prefix):
    return pd.Series([f"{prefix}_{i:05d}" for i in range(1, len(index) + 1)], index=index)


def as_packaged(frame, mapping):
    labels = mapping.loc[frame.index]
    if not frame.index.is_unique or not labels.is_unique or labels.isna().any():
        raise ValueError("Packaged stay keys must be unique and aligned")
    return frame.set_axis(pd.Index(labels.to_numpy(), name="stay_key"))


def table(folder, name, frame, members):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, compression="zstd")
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
    members[name] = {"kind": "parquet", "rows": len(frame), "columns": list(frame.columns), "sha256": sha(path)}


def arrays(folder, name, values, members):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    with np.load(path, allow_pickle=False) as restored:
        if sorted(restored.files) != sorted(values):
            raise ValueError("Unexpected fields in packaged array file")
        for key, value in values.items():
            np.testing.assert_array_equal(restored[key], value)
    members[name] = {
        "kind": "npz", "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in values.items()},
        "sha256": sha(path),
    }


def published_scores(directory, index, y, splits):
    outcomes = pd.concat(
        [pd.read_csv(directory / f"Outcomes-{split}.txt", index_col="RecordID", usecols=["RecordID", *SCORE_NAMES, DEATH])
         for split in splits], verify_integrity=True,
    ).loc[index]
    np.testing.assert_array_equal(y, outcomes[DEATH])
    scores = outcomes.loc[:, SCORE_NAMES]
    return scores.where(scores.ge(0))


def add_p12(folder, members):
    c_outcome, _, outcome_hashes = outcome_timing()
    ab, y_ab, folds = ab_inputs()
    c, y_c, _ = load_cohort()
    ab_ids, c_ids = keys(ab.index, "p12ab"), keys(c.index, "p12c")
    ab_labels = pd.read_csv(AB_LABELS, index_col="RecordID").loc[ab.index, ["source_set"]]
    ab_labels["in_hospital_death"] = y_ab.astype("int8")
    ab_labels["ab_fold"] = folds.astype("int8")
    c_labels = pd.DataFrame({"source_set": "C", "in_hospital_death": y_c.astype("int8")}, index=c.index)
    c_flags = pd.DataFrame(cohort_masks(c, c_outcome.loc[c.index], y_c), index=c.index).drop(columns="all")

    timing = pd.read_parquet(P22 / "features_AB_temporal.parquet")
    if not timing.index.equals(ab.index):
        raise ValueError("P22 A/B timing keys differ")

    def derive(records, index):
        last = records.groupby("RecordID").elapsed_minute.max().reindex(index).to_numpy()
        return time_summary(records, index), hourly_cube(records, index), last

    c_timing, c_cube, last = set_c_temporal(c.index, extension=derive)
    c_flags["last_sequence_observation_minute"] = last
    for split, base, labels, time, key, y in (
        ("ab", ab, ab_labels, timing, ab_ids, y_ab), ("c", c, c_labels, c_timing, c_ids, y_c),
    ):
        prefix = f"p12/{split}"
        table(folder, f"{prefix}_48h_features.parquet", as_packaged(base.join(time), key), members)
        table(folder, f"{prefix}_outcomes.parquet", as_packaged(labels, key), members)
        external = published_scores(DATA, base.index, y, ("a", "b") if split == "ab" else ("c",))
        table(folder, f"{prefix}_clinical_scores.parquet", as_packaged(external, key), members)
    table(folder, "p12/c_time_sensitivity.parquet", as_packaged(c_flags, c_ids), members)

    ab_seq = PREPARED / "ab.npz"
    manifest = from_json(PREPARED / "manifest.json")
    verify_sources(manifest["source_sha256"])
    if sha(ab_seq) != manifest["bundle_sha256"] or manifest["groups"] != list(GROUPS) or manifest["views"] != list(VIEWS):
        raise ValueError("P33 A/B tensor or channel order differs")
    with np.load(ab_seq, allow_pickle=False) as source:
        np.testing.assert_array_equal(source["y"], y_ab)
        np.testing.assert_array_equal(source["folds"], folds)
        values = {key: source[key] for key in ("cube", "static", "val_hidden", "val_static", "aug_hidden", "aug_static")}
    np.testing.assert_array_equal(values["static"], static_values(ab))
    arrays(folder, "p12/ab_48h_sequence.npz", {"stay_key": ab_ids.to_numpy(dtype=str), **values}, members)
    arrays(folder, "p12/c_48h_sequence.npz", {
        "stay_key": c_ids.to_numpy(dtype=str), "cube": c_cube, "static": static_values(c),
    }, members)

    positions, plan, mask_digest = plan_for(c)
    if from_json(REFERENCES / "manifest.json")["mask_draws_sha256"] != mask_digest:
        raise ValueError("P9 masking differs from the locked Set C experiment")
    draws = mask_draws(positions, plan)
    mask_arrays = {"stay_key": c_ids.to_numpy(dtype=str)}
    mask_arrays.update({f"full_{key}": rows for key, rows in positions.items()})
    mask_arrays.update({f"matched_{key}": np.stack([draw[key] for draw in draws]) for key in draws[0]})
    arrays(folder, "p12/c_48h_masks.npz", mask_arrays, members)

    ab24, time_ab24, y_ab24, folds24, required = development_24h()
    manifest24 = from_json(P49 / "manifest.json")
    if required != manifest24["source_sha256"] or (len(y_ab24), int(y_ab24.sum())) != (
        manifest24["ab_stays"], manifest24["ab_deaths"],
    ):
        raise ValueError("P49 A/B eligibility differs")
    c24, time_c24, y_c24, views24, _, counts24 = set_c_24h()
    historical24 = from_json(P49 / "evaluation" / "summary.json")["set_c"]
    if any(historical24[k] != value for k, value in counts24.items()):
        raise ValueError("P49 Set C eligibility differs")
    np.testing.assert_array_equal(y_ab24, ab_labels.loc[ab24.index, "in_hospital_death"])
    np.testing.assert_array_equal(folds24, ab_labels.loc[ab24.index, "ab_fold"])
    np.testing.assert_array_equal(y_c24, c_labels.loc[c24.index, "in_hospital_death"])
    table(folder, "p12/ab_24h_features.parquet", as_packaged(ab24.join(time_ab24), ab_ids), members)
    table(folder, "p12/c_24h_features.parquet", as_packaged(c24.join(time_c24), c_ids), members)
    channels = [col.removesuffix("__count") for col in c24 if col.endswith("__count")]
    mask24 = {"stay_key": c_ids.loc[c24.index].to_numpy(dtype=str)}
    for view, altered in views24.items():
        counts = [f"{channel}__count" for channel in channels]
        mask24[f"{view}_hidden"] = (c24[counts].gt(0) & altered[counts].eq(0)).to_numpy()
        mask24[f"{view}_icutype_hidden"] = (c24.static_ICUType.notna() & altered.static_ICUType.isna()).to_numpy()
    arrays(folder, "p12/c_24h_masks.npz", mask24, members)
    return {
        "ab_48h": [len(ab), int(y_ab.sum())], "c_48h": [len(c), int(y_c.sum())],
        "ab_24h": [len(ab24), int(y_ab24.sum())], "c_24h": [len(c24), int(y_c24.sum())],
        "outcomes_sha256": outcome_hashes, "c_48h_mask_sha256": mask_digest, "c_24h_mask_channels": channels,
        "sequence_views": list(VIEWS), "base_features": ab.columns.tolist(), "temporal_features": timing.columns.tolist(),
    }


def eicu_groups(cohort, fold_reference, group_reference):
    part, sizes, grouping = partition(cohort)
    if grouping != group_reference or len(sizes) != len(fold_reference):
        raise ValueError("Published Demo fold grouping differs")
    for got, expected in zip(sizes, fold_reference, strict=True):
        if any(expected[name] != value for name, value in got.items()):
            raise ValueError("Published Demo fold sizes differ")
    folds = np.full(len(cohort), -1, dtype=np.int8)
    for index, (_, test) in enumerate(part):
        folds[test] = index
    _, block_rows, _, _, _ = hospital_blocks(cohort)
    blocks = np.full(len(cohort), -1, dtype=np.int16)
    for i, rows in enumerate(block_rows):
        blocks[rows] = i + 1
    if (folds < 0).any() or (blocks < 1).any():
        raise ValueError("Unassigned Demo fold or site block")
    return folds, blocks


def demo_labels(cohort, patient_codes, site_codes, intended, grouping):
    folds, blocks = eicu_groups(cohort, intended, grouping)
    apache, included, coverage = demo_apache(cohort)
    apache_risk = np.full(len(cohort), np.nan)
    apache_risk[included] = apache
    label = pd.DataFrame({
        "in_hospital_death": cohort.y,
        "patient_key": cohort.uniquepid.map(patient_codes),
        "hospital_key": cohort.hospitalid.map(site_codes),
        "hospital_block": blocks, "demo_fold": folds, "apache_iva_hospital_risk": apache_risk,
    }, index=cohort.index)
    if label[["patient_key", "hospital_key"]].isna().any().any():
        raise ValueError("Missing exported Demo group key")
    return label, coverage


def add_demo(folder, members, p12):
    baseline = from_json(P28 / "summary.json")
    verify_sources(baseline["source_sha256"])
    cohort, cases, _, hashes = eicu_sequence_cases(baseline, p12["base_features"], p12["temporal_features"])
    cohort24, inputs24, flow, excluded, urine = demo_inputs_24h(p12["base_features"])
    earlier = from_json(P49 / "evaluation" / "summary.json")
    if (flow, excluded, urine) != (earlier["eicu_flow"], earlier["eicu_late_or_unknown_entry"], earlier["urine_coverage"]):
        raise ValueError("P49 Demo feature extraction differs")
    pd.testing.assert_frame_equal(cohort24.loc[cohort.index], cohort)
    demo_ids = keys(cohort24.index, "eicu")
    patients = {key: i for i, key in enumerate(pd.unique(cohort24.uniquepid), start=1)}
    sites = {key: i for i, key in enumerate(pd.unique(cohort24.hospitalid), start=1)}
    result48, result24 = from_json(P44 / "summary.json"), from_json(P50 / "summary.json")["demo"]
    labels48, coverage48 = demo_labels(cohort, patients, sites, result48["folds"], result48["grouping"])
    labels24, coverage24 = demo_labels(cohort24, patients, sites, result24["folds"], result24["grouping"])
    if coverage48 != earlier["eicu_48h_subset"]["cohort"] or coverage24 != earlier["eicu_landmark"]["cohort"]:
        raise ValueError("P49 APACHE comparison coverage differs")
    table(folder, "eicu_demo/outcomes_48h.parquet", as_packaged(labels48, demo_ids), members)
    table(folder, "eicu_demo/outcomes_24h.parquet", as_packaged(labels24, demo_ids), members)
    for alignment, mode, frame, cube in cases:
        if not frame.index.equals(cohort.index):
            raise ValueError("Demo sequence row order differs")
        prefix = f"eicu_demo/48h_{alignment}_{mode}"
        table(folder, f"{prefix}_features.parquet", as_packaged(frame, demo_ids), members)
        arrays(folder, f"{prefix}_sequence.npz", {
            "stay_key": demo_ids.loc[cohort.index].to_numpy(dtype=str), "cube": cube, "static": static_values(frame),
        }, members)
    for mode in MODES:
        base, time = inputs24[mode]
        table(folder, f"eicu_demo/24h_urine_mapped_{mode}_features.parquet",
              as_packaged(base.join(time), demo_ids), members)
    return {
        "48h": coverage48, "24h": coverage24,
        "raw_sha256": {**hashes, APACHE_FILE.name: APACHE_SHA256, "hospital.csv.gz": HOSPITAL_SHA256},
    }


def verify_archive(path, archive_members):
    with ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(archive_members) or archive.testzip() is not None:
            raise ValueError("Unexpected member or failed CRC check in dataset archive")
        for name, expected in archive_members.items():
            with archive.open(name) as content:
                actual = hashlib.file_digest(content, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Packaged member hash differs: {name}")


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = {p.relative_to(OUT).as_posix(): sha(p) for p in ARTIFACT_FILES}
    OUTPUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent, prefix="assemble-") as temporary:
        work = Path(temporary)
        data, staged = work / "data", work / OUTPUT.name
        data.mkdir()
        staged.mkdir()
        members = {}
        p12 = add_p12(data, members)
        print("Verified all P12 model inputs and reconstructed study masks", flush=True)
        demo = add_demo(data, members, p12)
        print("Verified both Demo horizons, group folds and clinical score coverage", flush=True)
        dictionary = pd.read_csv(OUT / "feature_dictionary.csv").set_index("feature")
        if list(dictionary.index) != p12["base_features"]:
            raise ValueError("The feature dictionary and base schema differ")
        table(data, "feature_dictionary_435.parquet", dictionary, members)
        (data / "README.txt").write_bytes(README.read_bytes())
        members["README.txt"] = {"kind": "text", "sha256": sha(data / "README.txt")}
        manifest = {
            "format_version": 1, "device": "cpu", "p12": p12, "eicu_demo": demo, "files": members,
            "artifact_source_sha256": source, "set_c_48h_scenarios": SCENARIOS,
            "sequence_channels": GROUPS, "sequence_hours": HOURS, "sequence_static_continuous": STATIC_NUM,
            "sequence_static_onehot": STATIC_CODES, "temporal_summary_statistics": STATS,
        }
        save_json(data / "manifest.json", manifest)
        verify_sources(source)
        if any(sha(DATA / f"Outcomes-{split}.txt") != expected for split, expected in OFFICIAL_SHA256.items()):
            raise ValueError("Official P12 outcomes changed during packaging")
        if any(sha(DEMO / name) != digest for name, digest in demo["raw_sha256"].items()):
            raise ValueError("Official Demo sources changed during packaging")
        exported = {k: item["sha256"] for k, item in members.items()}
        exported["manifest.json"] = sha(data / "manifest.json")
        archive = staged / ARCHIVE_NAME
        with ZipFile(archive, "w") as bundle:
            for name in sorted(exported):
                bundle.write(data / name, arcname=name)
        verify_archive(archive, exported)
        digest = sha(archive)
        (staged / "SHA256SUMS.txt").write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="utf-8")
        staged.rename(OUTPUT)
    print(f"Verified {len(exported)} archive members: {OUTPUT / ARCHIVE_NAME}", flush=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with threadpool_limits(limits=2):
        run()
