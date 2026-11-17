"""Fit P12-only 24-hour predictors and evaluate frozen models at the eICU landmark."""

import argparse
import hashlib
import os
import tarfile
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_p12_outcome_timing import DEATH, LOS, OFFICIAL_SHA256, SURVIVAL
from eicu_demo_features import DEMO, MODES, STAY, build_features, select_cohort
from evaluate_clinical_score_baselines import APACHE_FILE, APACHE_SHA256, demo_apache
from evaluate_eicu_demo import DEFAULT_OUTPUT as P8, source_hashes
from evaluate_eicu_urine_mapping import urine_events, urine_values
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import contrast_c
from evaluate_temporal_features import ARCHIVE, save_json
from pilot_temporal_features import GROUPS, ID, MINUTE, VALUE, time_summary
from prepare import DATA, OUT, extract_record, sha
from stress_missingness_c import C_FEATURES, C_LABELS, C_PREDICTIONS, load_cohort, score_metrics
from train_gcs_routed_temporal import TOLERANCE, VIEWS, build_views, gcs_full, matrix, prediction, score, train_one
from train_predictive_boosting import (
    AB_FEATURES, AB_FOLDS, AB_LABELS, FOLD_IDS, ROUNDS, SHIFT_SEED, ab_inputs, estimator,
    from_json, native_inputs, native_predict, verify_sources,
)

END_MINUTE = 1440
ROOT = OUT / "p49_24h_landmark"
ALIGNMENT = OUT / "record_alignment.csv"
P0 = OUT / "preparation_manifest.json"
SUMMARIES = "summary_xgb"
OBSERVED = ("temporal", "temporal_gcs60", "temporal_shift")
BLIND = "blind_lgb"


def p12_outcomes(tag):
    path = DATA / f"Outcomes-{tag}.txt"
    if sha(path) != OFFICIAL_SHA256[tag]:
        raise ValueError(f"Unexpected official P12 outcomes for {tag.upper()}")
    return pd.read_csv(path, usecols=[ID, LOS, SURVIVAL, DEATH], index_col=ID)


def p12_landmark(raw, outcome, y):
    values = outcome.loc[raw.index]
    np.testing.assert_array_equal(values[DEATH], y)
    survival = values[SURVIVAL]
    known = survival.ge(2) | (survival.eq(-1) & (y == 0))
    return (raw.static_Age.ge(18) & values[LOS].ge(2) & known).to_numpy()


def p12_records(rows, index, columns):
    features = {}
    observations = []
    for rid, content in rows:
        if rid in features:
            raise ValueError("Duplicate P12 encounter")
        feature, records, _ = extract_record(content, rid, end_minute=END_MINUTE)
        features[rid] = feature
        observations.extend(
            (stay, minute, channel, value) for stay, minute, channel, _, value, flag, _ in records
            if flag == "valid" and channel in GROUPS
        )
    if set(features) != set(index):
        raise ValueError("Missing or unexpected P12 encounter")
    base = pd.DataFrame.from_dict(features, orient="index").loc[index, columns]
    base.index.name = ID
    timing = pd.DataFrame(observations, columns=[ID, MINUTE, "variable", VALUE])
    if timing.empty or timing[MINUTE].max() != END_MINUTE:
        raise ValueError("Expected observations through the 24-hour endpoint")
    return base, time_summary(timing, index, end_minute=END_MINUTE)


def development_records(index):
    record_map = pd.read_csv(ALIGNMENT, index_col=ID).loc[index]
    for rid, source, digest in record_map[["source_set", "sha256"]].itertuples():
        path = DATA / f"set-{source.lower()}" / f"{rid}.txt"
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"A/B record differs from P0 in Set {source}")
        yield rid, content


def archive_records():
    expected = from_json(P0)["C_sha256"]
    if sha(ARCHIVE) != expected:
        raise ValueError("Unexpected P12 Set C archive")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            stem = Path(member.name).stem
            if member.isfile() and stem.isdigit():
                content = archive.extractfile(member)
                if content is None:
                    raise ValueError("Cannot read P12 Set C")
                yield int(stem), content.read()
    if sha(ARCHIVE) != expected:
        raise ValueError("Set C archive changed during extraction")


def fit_summary(x, y):
    model = estimator("xgb3", ROUNDS[-1]).set_params(n_jobs=2)
    model.fit(native_inputs(x, "full"), y)
    return model


def summary_risk(model, x):
    return native_predict(model, {"algorithm": "xgb3"}, native_inputs(x, "full"), ROUNDS[-1])


def use_route(primary, blind, missing, weight):
    return np.where(missing, (1 - weight) * primary + weight * blind, primary)


def development_risks(x, timing, y, folds):
    observed = {tag: {view: np.full(len(y), np.nan) for view in VIEWS} for tag in (SUMMARIES, *OBSERVED, BLIND)}
    missing = {view: np.zeros(len(y), dtype=bool) for view in VIEWS}
    for fold in FOLD_IDS:
        began = time.perf_counter()
        held = folds == fold
        valid_x = x.loc[held]
        views = build_views(valid_x, fold)
        inputs = {view: matrix(valid_x, altered, timing.loc[held]) for view, altered in views.items()}
        base = fit_summary(x.loc[~held], y[~held])
        for view, altered in views.items():
            observed[SUMMARIES][view][held] = summary_risk(base, altered)
            missing[view][held] = altered["GCS__count"].eq(0).to_numpy()
        for tag in (*OBSERVED, BLIND):
            model = train_one(tag, x.loc[~held], timing.loc[~held], y[~held], SHIFT_SEED + 100 * fold)
            for view, values in inputs.items():
                observed[tag][view][held] = prediction(tag, model, values)
        print(f"A/B fold {fold}, {time.perf_counter() - began:.1f}s", flush=True)
    return observed, missing


def select_candidate(y, folds, observed, missing):
    metrics = {SUMMARIES: score(y, folds, observed[SUMMARIES])}
    recipes = {SUMMARIES: {"primary": SUMMARIES, "blind_weight": 0}}
    for tag in OBSERVED:
        for weight in (0, 0.5, 1):
            key = f"{tag}:{weight}"
            recipes[key] = {"primary": tag, "blind_weight": weight}
            metrics[key] = score(y, folds, {
                view: use_route(observed[tag][view], observed[BLIND][view], missing[view], weight) for view in VIEWS
            })
    guard = {key: metrics[SUMMARIES][key] - TOLERANCE for key in ("clean", "full_gcs")}
    eligible = [key for key in recipes if all(metrics[key][view] >= value for view, value in guard.items())]
    selected = max(eligible, key=lambda key: (metrics[key]["objective"], metrics[key]["clean"]))
    return metrics, recipes, eligible, selected


def development_24h():
    input_files = (AB_FEATURES, AB_LABELS, AB_FOLDS, ALIGNMENT, P0)
    protected = {path.relative_to(OUT).as_posix(): sha(path) for path in input_files}
    raw, y, folds = ab_inputs()
    outcome = pd.concat([p12_outcomes(tag) for tag in ("a", "b")], verify_integrity=True)
    eligible = p12_landmark(raw, outcome, y)
    x, timing = p12_records(development_records(raw.index), raw.index, raw.columns)
    pd.testing.assert_frame_equal(x.filter(like="static_"), raw.filter(like="static_"), check_dtype=False)
    if not (x.filter(like="__count") <= raw.filter(like="__count")).all().all():
        raise ValueError("A/B feature counts increased after shortening the window")
    return x.loc[eligible], timing.loc[eligible], y[eligible], folds[eligible], protected


def fit():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    x, timing, y, folds, protected = development_24h()
    observed, missing = development_risks(x, timing, y, folds)
    metrics, recipes, passed, selected = select_candidate(y, folds, observed, missing)
    recipe = recipes[selected]
    built = {SUMMARIES: fit_summary(x, y)}
    if recipe["primary"] != SUMMARIES:
        tags = [recipe["primary"]]
        if recipe["blind_weight"] > 0:
            tags.append(BLIND)
        for tag in tags:
            built[tag] = train_one(tag, x, timing, y, SHIFT_SEED + 400)
    verify_sources(protected)
    for tag in ("a", "b"):
        p12_outcomes(tag)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p49-fitting-") as temporary:
        staged = Path(temporary) / ROOT.name
        staged.mkdir()
        hashes = {}
        for name, model in built.items():
            path = staged / f"{name}.joblib"
            joblib.dump(model, path)
            hashes[name] = sha(path)
        save_json(staged / "manifest.json", {
            "device": "cpu", "end_minute": END_MINUTE, "ab_stays": len(y), "ab_deaths": int(y.sum()),
            "ab_fold_sizes": [int((folds == fold).sum()) for fold in FOLD_IDS],
            "ab_fold_deaths": [int(y[folds == fold].sum()) for fold in FOLD_IDS],
            "cross_validation_fits": len(FOLD_IDS) * (len(OBSERVED) + 2), "final_fits": len(built),
            "ab_metrics": metrics, "recipes": recipes, "eligible": passed, "selected": selected,
            "model_sha256": hashes, "source_sha256": protected,
        })
        staged.rename(ROOT)
    print(f"Frozen 24-hour candidate from A/B: {selected}", flush=True)


def frozen():
    manifest = from_json(ROOT / "manifest.json")
    if manifest["device"] != "cpu" or manifest["end_minute"] != END_MINUTE:
        raise ValueError("Incorrect frozen 24-hour configuration")
    verify_sources(manifest["source_sha256"])
    models = {}
    for name, digest in manifest["model_sha256"].items():
        path = ROOT / f"{name}.joblib"
        if sha(path) != digest:
            raise ValueError("Frozen landmark model changed")
        models[name] = joblib.load(path)
    return models, manifest["recipes"][manifest["selected"]]


def model_risks(models, recipe, original, altered, timing):
    summary = summary_risk(models[SUMMARIES], altered)
    tag, weight = recipe["primary"], recipe["blind_weight"]
    if tag == SUMMARIES:
        return {SUMMARIES: summary, "selected": summary}
    values = matrix(original, altered, timing)
    selected = prediction(tag, models[tag], values)
    if weight:
        absent = altered["GCS__count"].eq(0).to_numpy()
        selected = use_route(selected, prediction(BLIND, models[BLIND], values), absent, weight)
    return {SUMMARIES: summary, "selected": selected}


def set_c_24h():
    paths = (C_FEATURES, C_LABELS, C_PREDICTIONS)
    hashes = {path.relative_to(OUT).as_posix(): sha(path) for path in paths}
    reference, y, historical = load_cohort()
    included = p12_landmark(reference, p12_outcomes("c"), y)
    base, timing = p12_records(archive_records(), reference.index, reference.columns)
    pd.testing.assert_frame_equal(base.filter(like="static_"), reference.filter(like="static_"), check_dtype=False)
    base, timing, target = base.loc[included], timing.loc[included], y[included]
    views = {"clean": base, "gcs_full": gcs_full(base, SHIFT_SEED), "mapped1": build_views(base, FOLD_IDS[0])["mapped1"]}
    verify_sources(hashes)
    metadata = {"stays": len(target), "deaths": int(target.sum()), "excluded_deaths": int(y[~included].sum()),
                "source_sha256": hashes}
    return base, timing, target, views, historical.loc[included], metadata


def set_c(models, recipe):
    base, timing, target, views, historical, metadata = set_c_24h()
    results = {}
    for key, frame in views.items():
        risk = model_risks(models, recipe, base, frame, timing)
        metrics = {tag: score_metrics(target, values) for tag, values in risk.items()}
        if key == "clean":
            for tag in ("Full", "Elastic"):
                metrics[f"frozen48_{tag}"] = score_metrics(target, historical[f"score_{tag}"].to_numpy())
        results[key] = {
            "metrics": metrics,
            "paired_ci": contrast_c(target, {SUMMARIES: risk[SUMMARIES]}, {"selected": risk["selected"]}),
        }
    return {**metadata, "views": results}


def temporal_extension(events, index):
    columns = {STAY: ID, "minute": MINUTE, "value": VALUE}
    return time_summary(events.rename(columns=columns), index, end_minute=END_MINUTE)


def demo_inputs_24h(columns):
    cohort, frames, counts, excluded = build_features(DEMO, temporal_extension, end_minute=END_MINUTE)
    events, urine_coverage = urine_events(cohort.index, end_minute=END_MINUTE)
    urine_base = urine_values(cohort.index, events)
    urine_timing = temporal_extension(events, cohort.index).filter(like="temporal_Urine__")
    inputs = {}
    for mode, frame in frames.items():
        frame.loc[:, urine_base.columns] = urine_base
        frame.loc[:, urine_timing.columns] = urine_timing
        inputs[mode] = (frame.loc[:, columns], frame.drop(columns=columns))
    return cohort, inputs, counts, excluded, urine_coverage


def demo_result(cohort, scores):
    apache, included, counts = demo_apache(cohort)
    subset = cohort.loc[included]
    full = {mode: {name: score_metrics(cohort.y.to_numpy(), values) for name, values in risks.items()}
            for mode, risks in scores.items()}
    matched = {mode: {name: values[included] for name, values in risks.items()} for mode, risks in scores.items()}
    metrics = {mode: {name: score_metrics(subset.y.to_numpy(), values)
                      for name, values in {"apache_iva": apache, **risks}.items()}
               for mode, risks in matched.items()}
    contrast_apache = paired_intervals(
        subset, dict.fromkeys(MODES, apache), {mode: values["selected"] for mode, values in matched.items()},
    )
    contrast_summary = paired_intervals(
        subset, {mode: values[SUMMARIES] for mode, values in matched.items()},
        {mode: values["selected"] for mode, values in matched.items()},
    )
    return {"cohort": counts, "all_metrics": full, "matched_metrics": metrics, "paired_apache": contrast_apache,
            "paired_summary": contrast_summary}


def evaluate():
    output = ROOT / "evaluation"
    if output.exists():
        raise FileExistsError(output)
    models, recipe = frozen()
    digest = sha(ROOT / "manifest.json")
    c = set_c(models, recipe)
    print("Scored locked models on the P12 Set C landmark proxy", flush=True)
    hashes = source_hashes(DEMO)
    if hashes != from_json(P8 / "summary.json")["data_sha256"]:
        raise ValueError("Unexpected eICU Demo release inputs")
    columns = pd.read_parquet(AB_FEATURES).columns
    cohort, inputs, counts, excluded, urine_coverage = demo_inputs_24h(columns)
    old_cohort, _, _ = select_cohort(DEMO)
    if not old_cohort.index.isin(cohort.index).all():
        raise ValueError("The 48-hour Demo cohort is not a subset")
    pd.testing.assert_frame_equal(cohort.loc[old_cohort.index], old_cohort)
    scores = {mode: model_risks(models, recipe, base, base, timing) for mode, (base, timing) in inputs.items()}
    landmark = demo_result(cohort, scores)
    contained = cohort.index.isin(old_cohort.index)
    old = demo_result(cohort.loc[contained], {
        mode: {name: values[contained] for name, values in risks.items()} for mode, risks in scores.items()
    })
    if source_hashes(DEMO) != hashes or sha(APACHE_FILE) != APACHE_SHA256 or sha(ROOT / "manifest.json") != digest:
        raise ValueError("Landmark inputs changed during evaluation")
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p49-evaluation-") as temporary:
        stage = Path(temporary) / "evaluation"
        stage.mkdir()
        save_json(stage / "summary.json", {
            "device": "cpu", "end_minute": END_MINUTE, "set_c": c, "eicu_landmark": landmark,
            "eicu_48h_subset": old, "eicu_flow": counts, "eicu_late_or_unknown_entry": excluded,
            "urine_coverage": urine_coverage, "model_manifest_sha256": digest,
            "raw_eicu_sha256": {**hashes, APACHE_FILE.name: APACHE_SHA256}, "raw_p12_sha256": OFFICIAL_SHA256,
        })
        stage.rename(output)
    print(f"Saved CPU landmark evaluation to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    options = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with threadpool_limits(limits=2):
        {"fit": fit, "evaluate": evaluate}[options.phase]()
