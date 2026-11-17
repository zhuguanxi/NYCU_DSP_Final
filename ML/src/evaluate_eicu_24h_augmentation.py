"""Test whether real eICU training stays improve the fixed 24-hour model."""

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from audit_eicu_hospital_variation import hospital_blocks
from eicu_demo_features import DEMO, MODES
from evaluate_24h_landmark import (
    APACHE_FILE, APACHE_SHA256, END_MINUTE, ROOT as P49, SUMMARIES, demo_apache, demo_inputs_24h,
    development_24h, frozen, model_risks, set_c_24h,
)
from evaluate_eicu_demo import source_hashes
from evaluate_gcs_routed_temporal import checked_metrics
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import contrast_c
from evaluate_real_eicu_augmentation import N_FOLDS, P26_SEED, partition, provenance
from evaluate_temporal_features import save_json
from prepare import OUT, sha
from stress_missingness_c import score_metrics
from train_gcs_routed_temporal import matrix, prediction, train_one
from train_predictive_boosting import from_json, verify_sources

OUTPUT = OUT / "p50_24h_eicu_augmentation"
BASE, UPDATED = "p49", "augmented"


def updated_risk(baseline, model, tag, raw, frame, timing):
    present = frame["GCS__count"].gt(0).to_numpy()
    updated = baseline.copy()
    if present.any():
        values = matrix(raw.loc[present], frame.loc[present], timing.loc[present])
        updated[present] = prediction(tag, model, values)
    np.testing.assert_array_equal(updated[~present], baseline[~present])
    return updated


def restrict(scores, included):
    return {mode: {name: score[included] for name, score in values.items()} for mode, values in scores.items()}


def metrics(cohort, scores):
    return {mode: {tag: score_metrics(cohort.y.to_numpy(), prob) for tag, prob in values.items()}
            for mode, values in scores.items()}


def pair(cohort, scores, control):
    return paired_intervals(
        cohort, {mode: values[control] for mode, values in scores.items()},
        {mode: values[UPDATED] for mode, values in scores.items()},
    )


def external(x, timing, y, models, recipe, reference):
    cohort, inputs, flow, excluded, urine = demo_inputs_24h(x.columns)
    if (flow, excluded, urine) != (
        reference["eicu_flow"], reference["eicu_late_or_unknown_entry"], reference["urine_coverage"],
    ):
        raise ValueError("24-hour cohort flow or raw feature coverage differs from P49")
    old = reference["eicu_landmark"]
    apache, included, counts = demo_apache(cohort)
    if counts != old["cohort"]:
        raise ValueError("24-hour evaluation population differs from P49")
    matched = cohort.loc[included]
    folds, sizes, grouping = partition(cohort)
    sources = provenance(cohort)
    raw_test = {mode: model_risks(models, recipe, base, base, time) for mode, (base, time) in inputs.items()}
    for mode, risks in raw_test.items():
        for tag, risk in risks.items():
            checked_metrics(cohort.y.to_numpy(), risk, old["all_metrics"][mode][tag])
            checked_metrics(matched.y.to_numpy(), risk[included], old["matched_metrics"][mode][tag])
        checked_metrics(matched.y.to_numpy(), apache, old["matched_metrics"][mode]["apache_iva"])
    scores = {mode: {BASE: values["selected"], UPDATED: np.full(len(cohort), np.nan)}
              for mode, values in raw_test.items()}
    extra_base, extra_timing = inputs["hourly"]
    extra = matrix(extra_base, extra_base, extra_timing)
    y_demo = cohort.y.to_numpy()
    tag = recipe["primary"]
    for index, (training, tested) in enumerate(folds, start=1):
        model = train_one(tag, x, timing, y, P26_SEED, extra=(extra[training], y_demo[training]))
        for mode, (frame, time) in inputs.items():
            scores[mode][UPDATED][tested] = updated_risk(
                scores[mode][BASE][tested], model, tag, frame.iloc[tested], frame.iloc[tested], time.iloc[tested],
            )
        sizes[index - 1]["train_deaths"] = int(y_demo[training].sum())
        sizes[index - 1]["test_apache_stays"] = int(included[tested].sum())
        print(f"Demo fold {index}/{N_FOLDS} complete, {len(training)} external training stays", flush=True)

    absent = extra_base["GCS__count"].eq(0).to_numpy()
    for frame, _ in inputs.values():
        np.testing.assert_array_equal(absent, frame["GCS__count"].eq(0))
    for values in scores.values():
        np.testing.assert_array_equal(values[UPDATED][absent], values[BASE][absent])
    common = restrict(scores, included)
    for values in common.values():
        values["apache_iva"] = apache
    _, blocks, _, _, _ = hospital_blocks(cohort)
    codes = np.empty(len(cohort), dtype=int)
    for i, members in enumerate(blocks):
        codes[members] = i
    site_bootstrap = cohort.assign(uniquepid=codes)
    site_common = site_bootstrap.loc[included]
    results = {
        "cohort": counts, "folds": sizes, "grouping": grouping, "provenance": sources,
        "all": {"metrics": metrics(cohort, scores), "vs_p49": pair(cohort, scores, BASE),
                "vs_p49_hospital_blocks": pair(site_bootstrap, scores, BASE)},
        "matched_apache": {
            "metrics": metrics(matched, common), "vs_p49": pair(matched, common, BASE),
            "vs_apache": pair(matched, common, "apache_iva"),
            "vs_p49_hospital_blocks": pair(site_common, common, BASE),
            "vs_apache_hospital_blocks": pair(site_common, common, "apache_iva"),
        },
        "gcs_absent_stays": int(absent.sum()), "gcs_absent_deaths": int(y_demo[absent].sum()),
    }
    return results, extra, y_demo


def evaluate_c(models, recipe, augmented, reference):
    base, timing, y, views, _, metadata = set_c_24h()
    if any(reference[name] != value for name, value in metadata.items()):
        raise ValueError("Set C provenance differs from P49")
    results = {}
    for name, frame in views.items():
        risks = model_risks(models, recipe, base, frame, timing)
        for label in (SUMMARIES, "selected"):
            checked_metrics(y, risks[label], reference["views"][name]["metrics"][label])
        old = risks["selected"]
        new = updated_risk(old, augmented, recipe["primary"], base, frame, timing)
        if name == "gcs_full":
            np.testing.assert_array_equal(old, new)
        results[name] = {
            "metrics": {BASE: score_metrics(y, old), UPDATED: score_metrics(y, new)},
            "paired_ci": contrast_c(y, {BASE: old}, {UPDATED: new})[UPDATED][BASE],
            "gcs_absent_stays": int(frame["GCS__count"].eq(0).sum()),
        }
    verify_sources(metadata["source_sha256"])
    return {**metadata, "views": results}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    models, recipe = frozen()
    if recipe != {"primary": "temporal", "blind_weight": 0.5}:
        raise ValueError("Expected the original 24-hour route")
    paths = (P49 / "manifest.json", P49 / "evaluation/summary.json")
    manifest, reference = map(from_json, paths)
    digests = {path.relative_to(OUT).as_posix(): sha(path) for path in paths}
    raw_hashes = {**source_hashes(DEMO), APACHE_FILE.name: sha(APACHE_FILE)}
    if (
        digests[paths[0].relative_to(OUT).as_posix()] != reference["model_manifest_sha256"]
        or raw_hashes != reference["raw_eicu_sha256"] or raw_hashes[APACHE_FILE.name] != APACHE_SHA256
    ):
        raise ValueError("The P49 manifest or Demo inputs differ from the frozen evaluation")
    x, timing, y, folds, protected = development_24h()
    if (
        len(y) != manifest["ab_stays"] or int(y.sum()) != manifest["ab_deaths"]
        or protected != manifest["source_sha256"] or reference["end_minute"] != END_MINUTE
    ):
        raise ValueError("Development data differ from P49")
    np.testing.assert_array_equal(np.bincount(folds), manifest["ab_fold_sizes"])
    print("Rebuilt P49 P12 training inputs", flush=True)
    demo, extra, y_demo = external(x, timing, y, models, recipe, reference)
    final = train_one(recipe["primary"], x, timing, y, P26_SEED, extra=(extra, y_demo))
    print("Fitted final 24-hour model on P12 A/B and the whole Demo training cohort", flush=True)
    c = evaluate_c(models, recipe, final, reference["set_c"])
    verify_sources({**protected, **digests})
    if (
        source_hashes(DEMO) != {key: value for key, value in raw_hashes.items() if key != APACHE_FILE.name}
        or sha(APACHE_FILE) != APACHE_SHA256
        or sha(DEMO / "hospital.csv.gz") != demo["provenance"]["hospital_sha256"]
    ):
        raise ValueError("External raw input changed during evaluation")
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p50-24h-") as temporary:
        stage = Path(temporary) / OUTPUT.name
        stage.mkdir()
        model_path = stage / "temporal.joblib"
        joblib.dump(final, model_path)
        save_json(stage / "summary.json", {
            "device": "cpu", "end_minute": END_MINUTE, "a_b_stays": len(y), "a_b_deaths": int(y.sum()),
            "fit_count": N_FOLDS + 1, "added_stay_weight": 1, "route": recipe, "demo": demo, "set_c": c,
            "source_sha256": {**protected, **digests}, "raw_eicu_sha256": raw_hashes,
            "model_sha256": sha(model_path),
        })
        stage.rename(OUTPUT)
    print(f"Saved 24-hour external augmentation comparisons to {OUTPUT}", flush=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with threadpool_limits(limits=2):
        run()
