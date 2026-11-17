"""Evaluate frozen models on a patient-disjoint, locally supplied eICU cohort."""

import argparse
import gzip
import hashlib
import tempfile
from pathlib import Path

from evaluate_gcs_routed_temporal import RoutedPredictor

import joblib
import pandas as pd

from select_robust_ensemble import OUTPUT as P28, blend

from eicu_demo_features import DEMO, INPUT_TABLES, STAY, build_features, read_table, select_cohort
from evaluate_eicu_urine_mapping import urine_events, urine_values
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import benchmarks, no_category_warning
from evaluate_temporal_features import FITTED as P22, TemporalPredictor, save_json
from pilot_temporal_features import ID, MINUTE, VALUE, time_summary
from prepare import OUT, sha
from stress_missingness_c import METRICS, score_metrics
from train_gcs_routed_temporal import BASELINE, OUTPUT as P26
from train_missingness_augmentation import from_json
from train_predictive_boosting import AB_FEATURES, FROZEN, verify_sources

PATIENT = "uniquepid"
SEED = "p27-eicu-confirmation-20261117"
CHUNK_SIZE = 200_000
BASELINES = ("temporal", "gcs_only_elastic")


def confirm_namespace(source):
    demo = read_table(DEMO, "patient", [STAY, PATIENT]).set_index(STAY)
    full = read_table(source, "patient", [STAY, PATIENT]).set_index(STAY)
    if not demo.index.is_unique or not full.index.is_unique or demo[PATIENT].isna().any():
        raise ValueError("Expected complete, unique Demo stay keys")
    overlap = full.reindex(demo.index)[PATIENT].astype("string")
    if overlap.isna().any() or not overlap.equals(demo[PATIENT].astype("string")):
        raise ValueError("The Demo patient-key namespace could not be verified in the full source")
    return set(demo[PATIENT].astype(str)), len(demo)


def verify_demo_patient_reference(expected):
    patient_path = DEMO / "patient.csv.gz"
    if sha(patient_path) != expected:
        raise ValueError("The Demo patient reference differs from the frozen P28 evaluation")


def choose_cohort(source, patients, smoke):
    cohort, _, _ = select_cohort(source)
    excluded, verified = (set(), 0) if smoke else confirm_namespace(source)
    series = cohort[PATIENT].astype(str)
    eligible = cohort.loc[~series.isin(excluded)]
    groups = sorted(
        set(eligible[PATIENT].astype(str)),
        key=lambda value: (hashlib.sha256(f"{SEED}:{value}".encode()).digest(), value),
    )[:patients]
    eligible = eligible.loc[eligible[PATIENT].astype(str).isin(groups)]
    if eligible.empty or eligible.y.nunique() != 2 or eligible[PATIENT].astype(str).isin(excluded).any():
        raise ValueError("Expected an eligible cohort of both outcomes and disjoint patients")
    return eligible, {"verified_demo_stays": verified, "excluded_demo_patients": len(excluded)}


def subset_tables(source, target, stays):
    counts = {}
    for name in INPUT_TABLES:
        file = source / f"{name}.csv.gz"
        wrote_header = False
        count = 0
        with gzip.open(target / file.name, "wt", encoding="utf-8", newline="") as handle:
            for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False):
                rows = chunk.loc[chunk[STAY].isin(stays)]
                if not wrote_header or len(rows):
                    rows.to_csv(handle, index=False, header=not wrote_header)
                    wrote_header = True
                    count += len(rows)
        if not wrote_header:
            pd.read_csv(file, nrows=0).to_csv(target / file.name, index=False, compression="gzip")
        counts[name] = count
    if counts["patient"] != len(stays):
        raise ValueError("Subset has a different number of patient records")
    return counts


def temporal_extension(events, index):
    return time_summary(events.rename(columns={STAY: ID, "minute": MINUTE, "value": VALUE}), index)


def mapped_inputs(directory, cohort, frames):
    events, coverage = urine_events(cohort.index, directory)
    urine = urine_values(cohort.index, events)
    timing = temporal_extension(events, cohort.index).filter(like="temporal_Urine__")
    for frame in frames.values():
        if frame["Urine__count"].ne(0).any():
            raise ValueError("Expected the original unmapped urine feature")
        frame.loc[:, urine.columns] = urine
        frame.loc[:, timing.columns] = timing
    return coverage


def frozen_models():
    p22_path, p26_path = P22 / "manifest.json", P26 / "manifest.json"
    p22, p26 = from_json(p22_path), from_json(p26_path)
    p22_model = P22 / "xgboost.joblib"
    if p22["device"] != "cpu" or p26["device"] != "cpu" or sha(p22_model) != p22["model_sha256"]:
        raise ValueError("Expected the frozen CPU models")
    protected = {
        **p22["source_sha256"], **p26["source_sha256"],
        **{p.relative_to(OUT).as_posix(): sha(p) for p in (p22_path, p26_path, p22_model)},
        **{(P26 / f"{tag}.joblib").relative_to(OUT).as_posix(): digest for tag, digest in p26["model_sha256"].items()},
    }
    verify_sources(protected)
    controls = benchmarks()
    choices = {BASELINE: controls[BASELINE], **{tag: joblib.load(P26 / f"{tag}.joblib") for tag in p26["model_sha256"]}}
    return (
        RoutedPredictor(p26["selected"], choices), TemporalPredictor(joblib.load(p22_model), p22["selected_rounds"]),
        controls["gcs_only_elastic"], p22["pilot"]["features"], protected,
    )


def frozen_ensemble(protected):
    selection_path = P28 / "manifest.json"
    evaluation_path = P28 / "evaluation" / "summary.json"
    selected = from_json(selection_path)
    if (
        selected != from_json(evaluation_path)["training"]
        or (selected["dataset"], selected["device"], selected["final_fits"]) != ("P12 A/B", "cpu", 0)
        or (selected["rule"] is not None and selected["rule"]["elastic"] != "gcs_only_elastic")
    ):
        raise ValueError("Expected the evaluated P28 CPU ensemble")
    for name, digest in selected["source_sha256"].items():
        if name in protected and protected[name] != digest:
            raise ValueError(f"Conflicting source digest for {name}")
        protected[name] = digest
    for path in (selection_path, evaluation_path):
        protected[path.relative_to(OUT).as_posix()] = sha(path)
    verify_sources(protected)
    pipe, _, columns = joblib.load(FROZEN / "Elastic.joblib")
    pipe.named_steps["model"].set_params(device="cpu", n_jobs=2)
    return selected["rule"], pipe, columns


def evaluate(source, patients, smoke, *, ensemble=False, sequence=None, adaptation=None):
    if sequence is not None and not ensemble:
        raise ValueError("Sequence evaluation requires the frozen P28 comparison")
    if adaptation is not None and sequence is None:
        raise ValueError("P46 adaptation evaluation includes the frozen P38 comparison")
    prefix = "p46" if adaptation is not None else "p40" if sequence is not None else "p29" if ensemble else "p27"
    output = OUT / f"{prefix}_{'demo_smoke' if smoke else 'independent_eicu'}"
    if output.exists():
        raise FileExistsError(output)
    if not smoke and source.samefile(DEMO):
        raise ValueError("Demo can only be used for verification")
    for table in INPUT_TABLES:
        if not (source / f"{table}.csv.gz").is_file():
            raise FileNotFoundError(source / f"{table}.csv.gz")
    new, temporal, gcs, extras, protected = frozen_models()
    demo_patient_sha = None
    if ensemble:
        rule, original_elastic, original_columns = frozen_ensemble(protected)
        p28_evaluation = from_json(P28 / "evaluation" / "summary.json")
        demo_patient_sha = p28_evaluation["eicu_raw_sha256"]["patient.csv.gz"]
        verify_demo_patient_reference(demo_patient_sha)
    if sequence is not None:
        for name, digest in sequence.source_sha256.items():
            if name in protected and protected[name] != digest:
                raise ValueError(f"Conflicting sequence source digest for {name}")
            protected[name] = digest
        verify_sources(protected)
    if adaptation is not None:
        if adaptation.demo_patient_sha256 != demo_patient_sha:
            raise ValueError("The P45 training patients differ from the pinned Demo reference")
        for name, digest in adaptation.source_sha256.items():
            if name in protected and protected[name] != digest:
                raise ValueError(f"Conflicting adaptation source digest for {name}")
            protected[name] = digest
        verify_sources(protected)
    candidate = "ensemble" if ensemble else "routed"
    baselines = ("frozen_elastic", "routed", *BASELINES) if ensemble else BASELINES
    input_sha = {table: sha(source / f"{table}.csv.gz") for table in INPUT_TABLES}
    original, overlap = choose_cohort(source, patients, smoke)
    if demo_patient_sha is not None:
        overlap["demo_patient_sha256"] = demo_patient_sha
    cols = pd.read_parquet(AB_FEATURES).columns.tolist() + extras
    with tempfile.TemporaryDirectory(dir=OUT, prefix="eicu-confirmation-") as temporary:
        stage = Path(temporary)
        copied = subset_tables(source, stage, original.index)
        extension = temporal_extension if sequence is None else sequence.features
        cohort, frames, _, _ = build_features(stage, feature_extension=extension)
        pd.testing.assert_frame_equal(cohort, original, check_dtype=False, check_exact=True)
        coverage = mapped_inputs(stage, cohort, frames)
        y = cohort.y.to_numpy()
        pipe, old_columns = gcs
        predicted = {name: {} for name in (candidate, *baselines)}
        for mode, frame in frames.items():
            predicted["routed"][mode] = new.predict_proba(frame.loc[:, cols])[:, 1]
            predicted["temporal"][mode] = temporal.predict_proba(frame.loc[:, cols])[:, 1]
            predicted["gcs_only_elastic"][mode] = no_category_warning(pipe, frame.loc[:, old_columns])
            if ensemble:
                predicted["frozen_elastic"][mode] = no_category_warning(original_elastic, frame.loc[:, original_columns])
                anchor = predicted["routed"][mode]
                predicted["ensemble"][mode] = anchor if rule is None else blend(
                    anchor, predicted["gcs_only_elastic"][mode], frame["GCS__count"].eq(0).to_numpy(), rule,
                )
        if sequence is not None:
            predicted["p38"] = sequence.predict(stage, frames, predicted["ensemble"])
            sequence_intervals = {
                name: paired_intervals(cohort, predicted[name], predicted["p38"])
                for name in ("ensemble", "frozen_elastic")
            }
        adaptation_checks = {}
        if adaptation is not None:
            adapted, adaptation_checks = adaptation.predict(frames, new, gcs, rule, cols, predicted["ensemble"])
            if not smoke:
                predicted["p45"] = adapted
                adaptation_intervals = {
                    name: paired_intervals(cohort, predicted[name], adapted)
                    for name in ("ensemble", "frozen_elastic", "p38")
                }
        intervals = {
            name: paired_intervals(cohort, predicted[name], predicted[candidate]) for name in baselines
        }
        result = []
        for mode in frames:
            stats = {name: score_metrics(y, group[mode]) for name, group in predicted.items()}
            row = {
                "mode": mode, "metrics": stats,
                "delta_vs_baseline": {
                    name: {metric: stats[candidate][metric] - stats[name][metric] for metric in METRICS}
                    for name in baselines
                },
                "paired_ci": {name: ci[mode] for name, ci in intervals.items()},
            }
            if sequence is not None:
                row["sequence_comparison"] = {
                    name: {
                        "delta": {metric: stats["p38"][metric] - stats[name][metric] for metric in METRICS},
                        "paired_ci": ci[mode],
                    } for name, ci in sequence_intervals.items()
                }
            if adaptation is not None and not smoke:
                row["adaptation_comparison"] = {
                    name: {
                        "delta": {metric: stats["p45"][metric] - stats[name][metric] for metric in METRICS},
                        "paired_ci": ci[mode],
                    } for name, ci in adaptation_intervals.items()
                }
            result.append(row)
        verify_sources(protected)
        if demo_patient_sha is not None:
            verify_demo_patient_reference(demo_patient_sha)
        if any(sha(source / f"{table}.csv.gz") != digest for table, digest in input_sha.items()):
            raise ValueError("The supplied eICU files changed during evaluation")
        deliverable = stage / "aggregate"
        deliverable.mkdir()
        summary = {
            "kind": "demo_runtime_test" if smoke else "patient_disjoint_eicu_confirmation",
            "device": "cpu", "fits": 0, "stays": len(cohort), "patients": int(cohort[PATIENT].nunique()),
            "candidate": candidate, "primary_baseline": baselines[0],
            "deaths": int(y.sum()), "sample_seed": SEED, "patient_cap": patients,
            "overlap_audit": overlap, "retained_source_rows": copied, "urine_coverage": coverage,
            "rows": result, "source_sha256": protected, "raw_eicu_sha256": input_sha,
        }
        if sequence is not None:
            summary["secondary_candidate"] = "p38"
            summary["sequence_recipe"] = sequence.recipe
        if adaptation is not None:
            summary["adaptation_candidate"] = "p45"
            summary["adaptation_recipe"] = adaptation.recipe
            summary["adaptation_checks"] = adaptation_checks
        save_json(deliverable / "summary.json", summary)
        deliverable.rename(output)
    print(f"Frozen confirmation pipeline completed in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-demo", action="store_true")
    modes.add_argument("--full-directory", type=Path)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--ensemble", action="store_true", help="Use the frozen P28 comparison (P29)")
    args = parser.parse_args()
    cap = args.max_patients if args.max_patients is not None else (120 if args.smoke_demo else 6000)
    if cap < 2:
        parser.error("--max-patients must be at least 2")
    evaluate(DEMO if args.smoke_demo else args.full_directory.resolve(), cap, args.smoke_demo, ensemble=args.ensemble)
