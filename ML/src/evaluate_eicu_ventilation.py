"""Audit timestamped GCS components and test observed ventilation in eICU Demo."""

import tempfile
from pathlib import Path

from evaluate_eicu_confirmation import frozen_ensemble, frozen_models, mapped_inputs

import numpy as np
import pandas as pd

from eicu_demo_features import DEMO, EVENT, STAY, normalize, p12_features, windowed
from evaluate_eicu_demo import MODES, source_hashes
from evaluate_gcs_routed_temporal import checked_metrics
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import no_category_warning
from evaluate_robust_ensemble import EVALUATION as P28, Ensemble
from evaluate_temporal_features import eicu_temporal, save_json
from pilot_temporal_features import OBSERVATIONS
from prepare import OUT, STATIC, sha
from stress_missingness_c import METRICS, score_metrics
from train_predictive_boosting import AB_FEATURES, from_json, verify_sources

OUTPUT = OUT / "p32_eicu_ventilation"
KEY = [STAY, "minute"]
PARTS = {"eyes": 4, "verbal": 5, "motor": 6}
EXPLICIT = {
    "o2 device": {"ventilator"},
    "oxygen delivery method": {"mechanical ventilator"},
    "mechanical ventilator mode": {"ac/cmv"},
    "ventilator support mode": {"cmv", "simv"},
}
OTHER_SUPPORT = {
    "nasal cannula", "high flow nasal cannula", "room air", "none (room air)",
    "bi-pap", "bipap", "cpap", "aerosol mask", "venturi mask", "high flow mask", "simple mask", "ambubag",
}
MODEL_NAMES = ("frozen_elastic", "gcs_only_elastic", "routed", "ensemble")


def gcs_audit(index):
    frame, _ = windowed(
        DEMO, "nurseCharting", index, "nursingchartoffset",
        ("nursingchartcelltypecat", "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"),
        "nursingchartentryoffset",
    )
    label = normalize(frame.nursingchartcelltypevallabel)
    name = normalize(frame.nursingchartcelltypevalname)
    total = (label.eq("glasgow coma score") & name.eq("gcs total")) | (
        label.eq("score (glasgow coma scale)") & name.eq("value")
    )
    reported = frame.loc[total, KEY].copy()
    reported["value"] = pd.to_numeric(frame.loc[total, "nursingchartvalue"], errors="coerce")
    valid = reported.value.between(3, 15) & reported.value.mod(1).eq(0)
    recorded = reported.loc[valid].groupby(KEY).value.min()

    scores = normalize(frame.nursingchartcelltypecat).eq("scores")
    parts = scores & label.eq("glasgow coma score") & name.isin(PARTS)
    rows = frame.loc[parts, KEY].copy()
    rows["part"] = name.loc[parts]
    points = pd.to_numeric(frame.loc[parts, "nursingchartvalue"], errors="coerce")
    rows["points"] = points.where(points.between(1, rows.part.map(PARTS)) & points.mod(1).eq(0))
    same_minute = rows.groupby([*KEY, "part"]).points.agg(["size", "count", "nunique", "min"])
    consistent = same_minute["size"].eq(same_minute["count"]) & same_minute["nunique"].eq(1)
    total_from_parts = same_minute.loc[consistent, "min"].unstack("part").reindex(columns=PARTS).dropna().sum(axis=1)

    overlap = recorded.index.intersection(total_from_parts.index)
    extra = total_from_parts.index.difference(pd.MultiIndex.from_frame(reported[KEY]))
    recorded_stays = recorded.index.get_level_values(STAY).unique()
    return {
        "reported_minutes": len(recorded), "reported_stays": len(recorded_stays),
        "complete_component_minutes": len(total_from_parts), "overlap": len(overlap),
        "equal_to_reported": int(total_from_parts.loc[overlap].eq(recorded.loc[overlap]).sum()),
        "conflicted_or_nonnumeric_total_minutes": len(
            total_from_parts.index.intersection(pd.MultiIndex.from_frame(reported.loc[~valid, KEY]))
        ),
        "extra_minutes": len(extra), "extra_stays": len(extra.get_level_values(STAY).unique().difference(recorded_stays)),
    }


def ventilation(index):
    frame, _ = windowed(
        DEMO, "respiratoryCharting", index, "respchartoffset",
        ("respcharttypecat", "respchartvaluelabel", "respchartvalue"),
        "respchartentryoffset",
    )
    cat = normalize(frame.respcharttypecat)
    label = normalize(frame.respchartvaluelabel)
    value = normalize(frame.respchartvalue)
    care = cat.eq("respflowcaredata")
    settings = cat.eq("respflowsettings")
    direct_source = {
        tag: frame.loc[care & label.eq(tag) & value.isin(allowed), KEY].drop_duplicates()
        for tag, allowed in EXPLICIT.items()
    }
    direct = pd.concat(direct_source.values(), ignore_index=True).drop_duplicates()
    numeric = pd.to_numeric(value, errors="coerce")
    positive = numeric.gt(0) & np.isfinite(numeric)
    volume = frame.loc[settings & label.eq("tidal volume (set)") & positive, KEY].drop_duplicates()
    rate = frame.loc[settings & label.eq("vent rate") & positive, KEY].drop_duplicates()
    paired = volume.merge(rate, on=KEY, validate="one_to_one")
    conflict = care & (
        (label.isin({"o2 device", "oxygen delivery method"}) & value.isin(OTHER_SUPPORT))
        | (label.eq("non-invasive ventilation mode") & value.notna() & value.ne(""))
    )
    conflicting = pd.MultiIndex.from_frame(frame.loc[conflict, KEY])
    raw = {"explicit": direct, "explicit_or_settings": pd.concat([direct, paired], ignore_index=True).drop_duplicates()}
    observations = {}
    coverage = {}
    for tag, entries in raw.items():
        selected = entries.loc[~pd.MultiIndex.from_frame(entries).isin(conflicting)]
        observations[tag] = selected.assign(variable="MechVent", value=1.0).loc[:, EVENT]
        coverage[tag] = {
            "minutes": len(selected), "stays": int(selected[STAY].nunique()),
            "conflicted_minutes_dropped": len(entries) - len(selected),
        }
    return observations, {
        "by_explicit_signal": {key: {"minutes": len(rows), "stays": int(rows[STAY].nunique())} for key, rows in direct_source.items()},
        "paired_setting_minutes": len(paired), "paired_setting_stays": int(paired[STAY].nunique()),
        "coverage": coverage,
    }


def evaluate_frames(frames, routed, ensemble, elastic, gcs):
    scores = {}
    for mode, frame in frames.items():
        scores[f"{mode}:routed"] = routed.predict_proba(frame)[:, 1]
        scores[f"{mode}:ensemble"] = ensemble.predict_proba(frame)[:, 1]
        for tag, (pipe, cols) in (("frozen_elastic", elastic), ("gcs_only_elastic", gcs)):
            scores[f"{mode}:{tag}"] = no_category_warning(pipe, frame.loc[:, cols])
    return scores


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    routed, _, gcs, extras, protected = frozen_models()
    rule, elastic_pipe, elastic_cols = frozen_ensemble(protected)
    ensemble = Ensemble(routed, gcs, rule)
    elastic = (elastic_pipe, elastic_cols)
    history = from_json(P28 / "summary.json")
    protected.update(history["source_sha256"])
    verify_sources(protected)
    vent = pd.read_parquet(OBSERVATIONS, columns=["variable", "clean_value", "quality_flag"])
    counts = vent.loc[vent.variable.eq("MechVent") & vent.quality_flag.eq("valid"), "clean_value"].value_counts()
    if counts.index.tolist() != [1.0]:
        raise ValueError("Historical A/B MechVent needs a positive-only observation mapping")

    (cohort, original), raw_hashes = eicu_temporal()
    if raw_hashes != history["eicu_raw_sha256"]:
        raise ValueError("The raw Demo files differ from P28")
    urine_coverage = mapped_inputs(DEMO, cohort, original)
    raw = pd.read_parquet(AB_FEATURES).columns.tolist()
    original = {mode: frame.loc[:, raw + extras] for mode, frame in original.items()}
    y = cohort.y.to_numpy()
    bank = gcs_audit(cohort.index)
    if any(int(frame["GCS__count"].gt(0).sum()) != bank["reported_stays"] for frame in original.values()):
        raise ValueError("GCS coverage differs from the P8 mapping")
    events, ventilation_coverage = ventilation(cohort.index)
    print(f"GCS source audit and ventilation coverage completed for {len(cohort)} stays", flush=True)
    reference = evaluate_frames(original, routed, ensemble, elastic, gcs)
    previous = {x["mode"]: x for x in history["eicu"] if x["alignment"] == "urine_mapped"}
    for mode in MODES:
        row = previous[mode]
        expected = {
            "ensemble": row["metrics"], "routed": row["comparison"]["routed"]["metrics"],
            "frozen_elastic": row["comparison"]["frozen_elastic_mapped"]["metrics"],
            "gcs_only_elastic": row["comparison"]["gcs_elastic_mapped"]["metrics"],
        }
        for tag, metrics in expected.items():
            checked_metrics(y, reference[f"{mode}:{tag}"], metrics)

    rows = [{
        "variant": "urine_only", "mode": mode,
        "metrics": {tag: score_metrics(y, reference[f"{mode}:{tag}"]) for tag in MODEL_NAMES},
    } for mode in MODES]
    static = pd.DataFrame(np.nan, index=cohort.index, columns=list(STATIC))
    for variant, observed in events.items():
        feature = p12_features(observed, static).filter(like="MechVent__")
        if int(feature.MechVent__count.sum()) != len(observed):
            raise ValueError("The ventilation event minutes did not survive feature construction")
        remapped = {mode: frame.copy() for mode, frame in original.items()}
        for frame in remapped.values():
            if frame.MechVent__count.ne(0).any():
                raise ValueError("Expected unmapped baseline MechVent")
            frame[feature.columns] = feature
        predicted = evaluate_frames(remapped, routed, ensemble, elastic, gcs)
        paired = paired_intervals(cohort, reference, predicted)
        baseline = {mode: predicted[f"{mode}:frozen_elastic"] for mode in MODES}
        revised = {mode: predicted[f"{mode}:ensemble"] for mode in MODES}
        same_input_comparison = paired_intervals(cohort, baseline, revised)
        for mode in MODES:
            metrics = {tag: score_metrics(y, predicted[f"{mode}:{tag}"]) for tag in MODEL_NAMES}
            old = {tag: score_metrics(y, reference[f"{mode}:{tag}"]) for tag in MODEL_NAMES}
            rows.append({
                "variant": variant, "mode": mode, "metrics": metrics,
                "changes": {
                    tag: {
                        "delta": {metric: metrics[tag][metric] - old[tag][metric] for metric in METRICS},
                        "paired_ci": paired[f"{mode}:{tag}"],
                    } for tag in MODEL_NAMES
                },
                "ensemble_vs_frozen_elastic": {
                    "delta": {metric: metrics["ensemble"][metric] - metrics["frozen_elastic"][metric] for metric in METRICS},
                    "paired_ci": same_input_comparison[mode],
                },
            })
        print(f"Completed frozen eICU evaluation with {variant} MechVent events", flush=True)

    if source_hashes(DEMO) != raw_hashes:
        raise ValueError("Raw eICU files changed during P32")
    verify_sources(protected)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p32-mapping-") as temporary:
        stage = Path(temporary) / OUTPUT.name
        stage.mkdir()
        save_json(stage / "summary.json", {
            "device": "cpu", "fits": 0, "stays": len(cohort), "patients": int(cohort.uniquepid.nunique()),
            "deaths": int(y.sum()), "p12_positive_vent_events": int(counts.iloc[0]),
            "gcs_audit": bank, "urine_coverage": urine_coverage, "ventilation_coverage": ventilation_coverage,
            "rows": rows, "source_sha256": protected, "eicu_raw_sha256": raw_hashes,
        })
        stage.rename(OUTPUT)


if __name__ == "__main__":
    run()
