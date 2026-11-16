"""Audit P12 time descriptors and score frozen Set C sensitivity cohorts."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from evaluate_eicu_confirmation import frozen_ensemble, frozen_models
from evaluate_gcs_routed_temporal import EVALUATION as P26_EVAL, RoutedPredictor, checked_metrics
from evaluate_hourly_sequence import ABLATION, GROUPS, ROOT as P33, hourly_cube, infer, static_values
from evaluate_ma_gbt_p12 import REFERENCES, plan_for
from evaluate_predictive_boosting import C_BOOTSTRAPS, C_BOOTSTRAP_SEED, benchmarks, no_category_warning
from evaluate_robust_ensemble import EVALUATION as P28_EVAL, Ensemble
from evaluate_temporal_features import save_json, set_c_temporal
from pilot_temporal_features import ID, MINUTE, time_summary
from prepare import DATA, OUT, sha
from prepare_hourly_sequence import hidden_channels
from sequence_blending import blend
from stress_missingness_c import C_FEATURES, C_LABELS, C_PREDICTIONS, CHANNELS, METRICS, apply_masks, load_cohort, load_pipelines, score_metrics
from train_predictive_boosting import AB_LABELS, from_json, verify_sources

OUTPUT = OUT / "p41_p12_outcome_timing"
OFFICIAL_SHA256 = {
    "a": "2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2",
    "b": "1305f29734da374a809483229a99f33a9230aa8d4fa74b1a2d6ad788136423d5",
    "c": "0b32dcb7753c0f18f2f5fde498a28618b8879c5e527ce0dae8ad120dd7e205c9",
}
LOS, SURVIVAL, DEATH = "Length_of_stay", "Survival", "In-hospital_death"
SCENARIOS = {"clean": (), "full_GCS": ("GCS",), "full_all_six": CHANNELS}
CONTRASTS = {
    "clean": (("p33", "frozen_full"), ("p38", "frozen_full")),
    "full_GCS": (("p26", "gcs_only_full"), ("p38", "frozen_full")),
}


def outcome_timing():
    tables, audit, hashes = {}, {}, {}
    for split in ("a", "b", "c"):
        path = DATA / f"Outcomes-{split}.txt"
        hashes[split] = sha(path)
        if hashes[split] != OFFICIAL_SHA256[split]:
            raise ValueError(f"Outcome source differs from PhysioNet v1.0.0 for set {split}")
        table = pd.read_csv(path, index_col="RecordID")
        if len(table) != 4000 or not table.index.is_unique or set(table[DEATH]) != {0, 1}:
            raise ValueError(f"Unexpected published set {split} outcomes")
        tables[split] = table
        flags = {
            "los_negative": table[LOS].lt(0),
            "los_day_0_1": table[LOS].isin((0, 1)),
            "los_day_2": table[LOS].eq(2),
            "survival_negative": table[SURVIVAL].lt(0),
            "survival_negative_except_minus_one": table[SURVIVAL].lt(-1),
            "survival_day_0_1": table[SURVIVAL].isin((0, 1)),
            "survival_day_2": table[SURVIVAL].eq(2),
            "either_day_0_1": table[LOS].isin((0, 1)) | table[SURVIVAL].isin((0, 1)),
        }
        audit[split] = {key: {"stays": int(flag.sum()), "deaths": int(table.loc[flag, DEATH].sum())}
                        for key, flag in flags.items()}
    ab = pd.read_csv(AB_LABELS, index_col="RecordID")
    for split in ("a", "b"):
        y = ab.loc[ab.source_set.eq(split.upper()), "In_hospital_death"]
        if set(y.index) != set(tables[split].index):
            raise ValueError(f"Development keys differ from the original set {split}")
        np.testing.assert_array_equal(y, tables[split].loc[y.index, DEATH])
    return tables["c"], audit, hashes


def cohort_masks(raw, timing, y):
    adult = raw.static_Age.ge(18).to_numpy()
    early = timing[LOS].isin((0, 1)).to_numpy() | timing[SURVIVAL].isin((0, 1)).to_numpy()
    day3 = timing[LOS].ge(3).to_numpy()
    survival = timing[SURVIVAL]
    known_late = survival.ge(3).to_numpy() | (survival.eq(-1).to_numpy() & (y == 0))
    return {
        "all": np.ones(len(y), dtype=bool),
        "adult": adult,
        "adult_exclude_day_0_1": adult & ~early,
        "adult_day3_sensitivity": adult & day3 & known_late,
    }


def ap_intervals(y, predictions, comparisons):
    names = sorted({name for pair in comparisons for name in pair})
    rng = np.random.default_rng(C_BOOTSTRAP_SEED)
    samples = {f"{new}_vs_{old}": [] for new, old in comparisons}
    for _ in range(C_BOOTSTRAPS):
        drawn = rng.integers(len(y), size=len(y))
        if y[drawn].min() == y[drawn].max():
            raise ValueError("Expected both outcomes in each Set C bootstrap draw")
        values = {name: average_precision_score(y[drawn], predictions[name][drawn]) for name in names}
        for new, old in comparisons:
            samples[f"{new}_vs_{old}"].append(values[new] - values[old])
    return {name: np.quantile(draws, (0.025, 0.975)).tolist() for name, draws in samples.items()}


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    timing, audit, outcome_hashes = outcome_timing()
    raw, y, historical = load_cohort()
    if set(raw.index) != set(timing.index) or {LOS, SURVIVAL, DEATH}.intersection(raw.columns):
        raise ValueError("Unexpected Set C keys or model schema")
    timing = timing.loc[raw.index]
    np.testing.assert_array_equal(y, timing[DEATH])
    cohorts = cohort_masks(raw, timing, y)
    positions, plan, mask_sha = plan_for(raw)
    reference_path = REFERENCES / "manifest.json"
    if from_json(reference_path)["mask_draws_sha256"] != mask_sha:
        raise ValueError("Set C mask positions differ from the frozen analysis")
    cases = {
        name: apply_masks(raw, {key: positions[key] for key in channels}, plan, "full") if channels else raw
        for name, channels in SCENARIOS.items()
    }

    routed, _, gcs, _, protected = frozen_models()
    rule, _, _ = frozen_ensemble(protected)
    original_pipes, original_hashes = load_pipelines(raw)
    full_pipe, full_cols = original_pipes["Full"]
    elastic_pipe, elastic_cols = original_pipes["Elastic"]
    gcs_pipe, gcs_cols = benchmarks()["gcs_only_full"]
    histories = {"p26": P26_EVAL, "p28": P28_EVAL, "p33": P33 / "evaluation", "p38": ABLATION / "evaluation"}
    paths = {tag: directory / "summary.json" for tag, directory in histories.items()}
    history = {tag: from_json(path) for tag, path in paths.items()}
    for summary in history.values():
        for key, digest in summary["source_sha256"].items():
            if key in protected and protected[key] != digest:
                raise ValueError(f"Conflicting evaluation source digest for {key}")
            protected[key] = digest
    for path in (C_FEATURES, C_LABELS, C_PREDICTIONS, AB_LABELS, reference_path, *paths.values()):
        protected[path.relative_to(OUT).as_posix()] = sha(path)
    for name, digest in original_hashes.items():
        protected[f"final_models/{name}.joblib"] = digest
    verify_sources(protected)
    p38 = history["p38"]["training"]
    p33 = history["p33"]["training"]
    included = [group for group in GROUPS if group not in {"Urine", "RespRate"}]
    if (
        p38["included_channels"] != included or p38["device"] != "cpu" or p38["block_hours"] != 4
        or p33["selected"] != ["plain", 8, 0.25] or p33["device"] != "cpu"
    ):
        raise ValueError("Unexpected P33 or P38 sequence configuration")
    roots = {"p33": P33, "p38": ABLATION}
    models = {key: root / "fitted/sequence.pt" for key, root in roots.items()}
    if any(sha(models[key]) != history[key]["training"]["model_sha256"] for key in roots):
        raise ValueError("A frozen sequence model differs from its evaluation")
    channels = [GROUPS.index(group) for group in included]

    def derive(records, index):
        last = records.groupby(ID)[MINUTE].max().reindex(index)
        return time_summary(records, index), hourly_cube(records, index), last

    timing_features, cubes, last = set_c_temporal(raw.index, extension=derive)
    hidden = np.stack([hidden_channels(raw, frame) for frame in cases.values()])
    routed_c = RoutedPredictor(routed.rule, routed.models, raw, timing_features)
    anchor = Ensemble(routed_c, gcs, rule)
    full = {key: {row["scenario"]: row for row in ref["set_c"]} for key, ref in history.items()}
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent, prefix="p41-timing-") as tmp:
        stage = Path(tmp)
        batch = {"cubes": cubes, "statics": static_values(raw), "hidden": hidden}
        hourly = infer(stage, models["p33"], **batch)
        reduced = infer(stage, models["p38"], four_hour=True, channels=channels, **batch)
        scores = {}
        for i, (case, frame) in enumerate(cases.items()):
            table = anchor.predict_proba(frame)[:, 1]
            predictions = {
                "frozen_full": no_category_warning(full_pipe, frame.loc[:, full_cols]),
                "frozen_elastic": no_category_warning(elastic_pipe, frame.loc[:, elastic_cols]),
                "gcs_only_full": no_category_warning(gcs_pipe, frame.loc[:, gcs_cols]),
                "p26": routed_c.predict_proba(frame)[:, 1],
                "p28": table,
                "p33": 0.75 * table + 0.25 * hourly[i].astype(float),
                "p38": blend(table, reduced[i], p38["selected"]),
            }
            for tag in full:
                checked_metrics(y, predictions[tag], full[tag][case])
            for tag in ("frozen_full", "gcs_only_full"):
                checked_metrics(y, predictions[tag], {m: full["p26"][case][f"{m}_{tag}"] for m in METRICS})
            if case == "clean":
                for tag, name in (("frozen_full", "Full"), ("frozen_elastic", "Elastic")):
                    np.testing.assert_array_equal(predictions[tag], historical[f"score_{name}"].to_numpy(dtype=predictions[tag].dtype))
            scores[case] = predictions
        print("Verified all three Set C views against the existing frozen metrics", flush=True)
        stats, intervals = {}, {}
        for name, take in cohorts.items():
            group = y[take]
            stats[name], intervals[name] = {}, {}
            for scenario, predictions in scores.items():
                filtered = {key: value[take] for key, value in predictions.items()}
                stats[name][scenario] = {tag: score_metrics(group, value) for tag, value in filtered.items()}
                if scenario in CONTRASTS:
                    intervals[name][scenario] = ap_intervals(group, filtered, CONTRASTS[scenario])
            print(f"P41 {name}: {len(group)} stays, {int(group.sum())} deaths", flush=True)
        early_death = timing[SURVIVAL].isin((0, 1)) & timing[DEATH].eq(1)
        last = last[early_death]
        clock_check = {"day_0_1_deaths": int(early_death.sum()), "after_24h": int(last.gt(1440).sum()),
                       "at_or_after_36h": int(last.ge(2160).sum())}
        verify_sources(protected)
        if any(sha(DATA / f"Outcomes-{name}.txt") != h for name, h in outcome_hashes.items()):
            raise ValueError("Outcome inputs changed during evaluation")
        public = stage / OUTPUT.name
        public.mkdir()
        save_json(public / "summary.json", {
            "device": "cpu", "fits": 0, "outcome_sha256": outcome_hashes, "time_audit": audit,
            "day_0_1_death_observation_clock": clock_check, "mask_draws_sha256": mask_sha,
            "cohorts": {tag: {"stays": int(sum(take)), "deaths": int(y[take].sum())} for tag, take in cohorts.items()},
            "metrics": stats, "paired_ap_ci": intervals, "repeats": C_BOOTSTRAPS, "seed": C_BOOTSTRAP_SEED,
            "source_sha256": protected,
        })
        public.rename(OUTPUT)
    print(f"P41 frozen CPU replay saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    evaluate()
