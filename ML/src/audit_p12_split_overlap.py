"""Check P12 split integrity beyond original RecordID equality."""

import csv
import hashlib
import json
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import DATA, OUT, UNITS, extract_record, sha

OUTPUT = OUT / "p47_p12_split_overlap"
C = OUT / "C_evaluation"
ARCHIVE = DATA / "set-c.tar.gz"
SPLITS = ("A", "B", "C")
COMPARISONS = (("A", "B"), ("A", "C"), ("B", "C"))
MIN_EVENTS = 40
MIN_VARIABLES = 3
MIN_MINUTES = 12


def read_inputs():
    alignment_ab = pd.read_csv(OUT / "record_alignment.csv", index_col="RecordID")
    alignment_c = pd.read_csv(C / "record_alignment_C.csv", index_col="RecordID")
    labels_ab = pd.read_csv(OUT / "labels_AB.csv", index_col="RecordID")
    labels_c = pd.read_csv(C / "labels_C.csv", index_col="RecordID")
    ab = pd.read_parquet(OUT / "features_AB.parquet")
    c = pd.read_parquet(C / "features_C.parquet")
    for alignment, labels, values in ((alignment_ab, labels_ab, ab), (alignment_c, labels_c, c)):
        if not alignment.index.is_unique or not labels.index.equals(values.index) or set(alignment.index) != set(values.index):
            raise ValueError("Alignment, label and feature keys differ")
    if set(ab.index).intersection(c.index) or list(ab.columns) != list(c.columns):
        raise ValueError("A/B and C overlap or have distinct schemas")
    if not labels_ab.source_set.equals(alignment_ab.loc[ab.index, "source_set"]):
        raise ValueError("Expected each A/B stay to have its original split")
    if (ab.shape, c.shape) != ((8000, 435), (4000, 435)):
        raise ValueError("Unexpected original study dimensions")
    labels = pd.concat((labels_ab, labels_c.assign(source_set="C")))
    for split in SPLITS:
        records = labels.loc[labels.source_set.eq(split), "In_hospital_death"]
        outcomes = pd.read_csv(DATA / f"Outcomes-{split.lower()}.txt", index_col="RecordID")["In-hospital_death"]
        if not outcomes.index.is_unique or set(outcomes.index) != set(records.index):
            raise ValueError(f"Original outcome keys differ for Set {split}")
        np.testing.assert_array_equal(records, outcomes.loc[records.index])
    return alignment_ab, alignment_c, labels, pd.concat((ab, c))


def payloads(alignment_ab, alignment_c):
    for split in ("A", "B"):
        expected = alignment_ab.loc[alignment_ab.source_set.eq(split)]
        paths = sorted((DATA / f"set-{split.lower()}").glob("*.txt"))
        if {int(path.stem) for path in paths} != set(expected.index):
            raise ValueError(f"Original local Set {split} files do not match their alignment")
        for path in paths:
            stay = int(path.stem)
            yield stay, split, path.read_bytes(), expected.at[stay, "sha256"]
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            stem = Path(member.name).stem
            if not member.isfile() or not stem.isdigit():
                continue
            stay = int(stem)
            if member.name != alignment_c.at[stay, "archive_member"]:
                raise ValueError("Unexpected archived Set C member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Could not extract an original Set C record")
            yield stay, "C", stream.read(), alignment_c.at[stay, "sha256"]


def event_tokens(records, token_ids):
    events = {
        (minute, variable, value)
        for _, minute, variable, _, value, flag, _ in records
        if flag == "valid" and variable in UNITS
    }
    eligible = (
        len(events) >= MIN_EVENTS
        and len({var for _, var, _ in events}) >= MIN_VARIABLES
        and len({minute for minute, _, _ in events}) >= MIN_MINUTES
    )
    return frozenset(token_ids.setdefault(event, len(token_ids)) for event in events), eligible


def signatures(alignment_ab, alignment_c, labels, features):
    tokens, eligible, token_ids, text, trace, feature = {}, {}, {}, {}, {}, {}
    for stay, split, contents, expected in payloads(alignment_ab, alignment_c):
        if stay in tokens or labels.at[stay, "source_set"] != split or hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError("A source payload disagrees with its original study alignment")
        _, events, _ = extract_record(contents, stay)
        values, include = event_tokens(events, token_ids)
        canonical_rows = sorted(
            (row["Time"].strip(), row["Parameter"].strip(), row["Value"].strip())
            for row in csv.DictReader(contents.decode().splitlines()) if row["Parameter"].strip() != "RecordID"
        )
        text[stay] = hashlib.sha256(json.dumps(canonical_rows, separators=(",", ":")).encode()).digest()
        if values:
            trace[stay] = hashlib.sha256(np.asarray(sorted(values), dtype="<u8").tobytes()).digest()
        tokens[stay], eligible[stay] = values, include
    if set(tokens) != set(labels.index):
        raise ValueError("The raw split audit did not cover all stays")
    rows = features.to_numpy(dtype="<f8", copy=True)
    rows[rows == 0] = 0
    rows[np.isnan(rows)] = np.nan
    for stay, row in zip(features.index, rows, strict=True):
        feature[stay] = hashlib.sha256(row.tobytes()).digest()
    return {"record_content": text, "nonempty_dynamic_observations": trace, "frozen_feature_values": feature}, tokens, eligible


def match_counts(pairs, labels):
    n, conflicting, left, right = 0, 0, set(), set()
    for a, b in pairs:
        n += 1
        left.add(a)
        right.add(b)
        conflicting += labels.at[a, "In_hospital_death"] != labels.at[b, "In_hospital_death"]
    return {
        "pairs": n, "left_stays": len(left), "right_stays": len(right), "conflicting_outcomes": int(conflicting),
    }


def exact_pairs(keys, left, right):
    index = defaultdict(list)
    for stay in left:
        if stay in keys:
            index[keys[stay]].append(stay)
    for other in right:
        if other in keys:
            for stay in index.get(keys[other], ()):
                yield stay, other


def prefixes(tokens):
    frequencies = Counter(token for values in tokens.values() for token in values)
    output = {}
    for stay, values in tokens.items():
        # An overlap of at least ceil(0.9 * len(values)) is necessary for 90% Jaccard.
        count = len(values) - (9 * len(values) + 9) // 10 + 1
        output[stay] = sorted(values, key=lambda token: (frequencies[token], token))[:count]
    return output


def high_overlap_pairs(tokens, prefix, left, right, stats):
    index = defaultdict(list)
    for stay in left:
        for token in prefix[stay]:
            index[token].append(stay)
    for other in right:
        right_tokens = tokens[other]
        for stay in {stay for token in prefix[other] for stay in index.get(token, ())}:
            left_tokens = tokens[stay]
            if 10 * min(len(left_tokens), len(right_tokens)) < 9 * max(len(left_tokens), len(right_tokens)):
                continue
            stats["candidate_pairs_tested"] += 1
            shared = len(left_tokens & right_tokens)
            union = len(left_tokens) + len(right_tokens) - shared
            if 10 * shared >= 9 * union:
                yield stay, other


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = OUT / "preparation_manifest.json"
    files = (
        manifest, OUT / "record_alignment.csv", OUT / "labels_AB.csv", OUT / "features_AB.parquet",
        C / "evaluation_manifest.json", C / "record_alignment_C.csv", C / "labels_C.csv", C / "features_C.parquet",
    )
    source_hashes = {path.relative_to(OUT).as_posix(): sha(path) for path in files}
    reference = json.loads((C / "evaluation_manifest.json").read_text(encoding="utf-8"))
    preparation = json.loads(manifest.read_text(encoding="utf-8"))
    c_sha = preparation["C_sha256"]
    outcome_hashes = {tag: sha(DATA / f"Outcomes-{tag}.txt") for tag in ("a", "b", "c")}
    if (
        (sha(ARCHIVE), reference["C_archive_sha256"]) != (c_sha, c_sha)
        or source_hashes["features_AB.parquet"] != reference["AB_feature_sha256"]
        or source_hashes["C_evaluation/features_C.parquet"] != reference["C_feature_sha256"]
        or outcome_hashes != {**preparation["input_label_sha256"], "c": reference["labels_sha256"]}
    ):
        raise ValueError("Expected the data used for the frozen Set C evaluation")
    alignment_ab, alignment_c, labels, features = read_inputs()
    keys, events, eligible = signatures(alignment_ab, alignment_c, labels, features)
    by_split = {tag: labels.index[labels.source_set.eq(tag)].tolist() for tag in SPLITS}
    kept = {tag: [stay for stay in values if eligible[stay]] for tag, values in by_split.items()}
    tokens = {stay: values for stay, values in events.items() if eligible[stay]}
    prefix = prefixes(tokens)
    summary = {}
    for a, b in COMPARISONS:
        stats = {"candidate_pairs_tested": 0, "possible_eligible_pairs": len(kept[a]) * len(kept[b])}
        for name, signature in keys.items():
            stats[name] = match_counts(exact_pairs(signature, by_split[a], by_split[b]), labels)
        stats["dynamic_jaccard_at_least_0_90"] = match_counts(
            high_overlap_pairs(tokens, prefix, kept[a], kept[b], stats), labels,
        )
        summary[f"{a}_vs_{b}"] = stats
        print(f"P47 {a} vs {b}: completed exact and high-overlap checks", flush=True)
    if (
        sha(ARCHIVE) != c_sha or any(sha(OUT / key) != value for key, value in source_hashes.items())
        or any(sha(DATA / f"Outcomes-{key}.txt") != value for key, value in outcome_hashes.items())
    ):
        raise ValueError("An original evaluation source changed")
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p12-overlap-") as temporary:
        stage = Path(temporary) / OUTPUT.name
        stage.mkdir()
        result = {
            "device": "cpu", "fits": 0, "by_split": {
                tag: {"stays": len(stays), "eligible_for_jaccard": len(kept[tag]),
                      "empty_dynamic_observations": sum(not events[stay] for stay in stays)}
                for tag, stays in by_split.items()
            },
            "jaccard": {"min_overlap": 0.9, "min_events": MIN_EVENTS, "min_variables": MIN_VARIABLES,
                        "min_minutes": MIN_MINUTES},
            "comparisons": summary, "set_c_archive_sha256": c_sha, "source_sha256": source_hashes,
            "outcome_sha256": outcome_hashes,
        }
        (stage / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        stage.rename(OUTPUT)
    print(f"P47 split overlap audit saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    run()
