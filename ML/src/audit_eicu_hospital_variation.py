"""Audit frozen eICU gains using hospital blocks and within-hospital ranking."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score

from eicu_demo_features import DEMO
from evaluate_eicu_demo import BOOTSTRAPS, SEED, ranking, source_hashes
from evaluate_frozen_calibration import ROOT as P42, calibrate, demo_scores
from evaluate_hourly_sequence import ABLATION
from evaluate_robust_ensemble import EVALUATION as P28
from evaluate_temporal_features import save_json
from prepare import OUT, sha
from train_predictive_boosting import AB_FEATURES, from_json, verify_sources

OUTPUT = OUT / "p43_eicu_hospital_variation"
MODELS = ("frozen_elastic", "p28", "p38")
PAIRS = {
    "p28_vs_frozen_elastic": ("p28", "frozen_elastic"),
    "p38_vs_p28": ("p38", "p28"),
    "p38_vs_frozen_elastic": ("p38", "frozen_elastic"),
}
CALIBRATED = ("p28", "p38")


def hospital_blocks(cohort):
    hospitals, sites = pd.factorize(cohort.hospitalid, sort=False)
    patients, people = pd.factorize(cohort.uniquepid, sort=False)
    if (hospitals < 0).any() or (patients < 0).any():
        raise ValueError("Hospital or patient keys are missing")
    incidence = coo_matrix(
        (np.ones(len(cohort), dtype=np.int32), (hospitals, patients)), shape=(len(sites), len(people)),
    ).tocsr()
    n_blocks, links = connected_components(incidence @ incidence.T, directed=False)
    blocks = links[hospitals]
    site_rows = [np.flatnonzero(hospitals == i) for i in range(len(sites))]
    block_rows = [np.flatnonzero(blocks == i) for i in range(n_blocks)]
    if pd.DataFrame({"patient": patients, "block": blocks}).groupby("patient").block.nunique().gt(1).any():
        raise ValueError("Patient split across hospital blocks")
    patient_rows = list(cohort.groupby("uniquepid", sort=False).indices.values())
    metadata = {
        "hospitals": len(sites), "patients": len(people), "hospital_blocks": n_blocks,
        "multi_hospital_patients": int((incidence.astype(bool).getnnz(axis=0) > 1).sum()),
        "hospital_size_min": int(min(map(len, site_rows))),
        "hospital_size_median": float(np.median(list(map(len, site_rows)))),
        "hospital_size_max": int(max(map(len, site_rows))),
        "hospital_block_size_max": int(max(map(len, block_rows))),
    }
    return site_rows, block_rows, links, patient_rows, metadata


def within_hospital(y, scores, sites, links, n_blocks):
    denominators = np.zeros(n_blocks, dtype=np.int64)
    contributions = {name: np.zeros(n_blocks) for name in MODELS}
    hospitals_with_pairs = 0
    for site, block in zip(sites, links, strict=True):
        values = y[site]
        deaths = int(values.sum())
        pairs = deaths * (len(values) - deaths)
        if not pairs:
            continue
        denominators[block] += pairs
        hospitals_with_pairs += 1
        for tag in MODELS:
            contributions[tag][block] += pairs * roc_auc_score(values, scores[tag][site])
    return denominators, contributions, hospitals_with_pairs


def difference(values):
    return {key: np.subtract(values[new], values[old]) for key, (new, old) in PAIRS.items()}


def bootstrap(y, scores, groups, *, within=None, errors=None):
    rng = np.random.default_rng(SEED)
    ap_auc = {tag: [] for tag in PAIRS}
    conditional = {tag: [] for tag in PAIRS}
    brier = {tag: [] for tag in CALIBRATED}
    for _ in range(BOOTSTRAPS):
        drawn = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in drawn])
        truth = y[idx]
        if truth.min() == truth.max():
            continue
        for tag, value in difference({name: ranking(truth, scores[name][idx]) for name in MODELS}).items():
            ap_auc[tag].append(value)
        if within is not None:
            denominators, contributions = within
            pairs = denominators[drawn].sum()
            if pairs:
                ranks = {tag: row[drawn].sum() / pairs for tag, row in contributions.items()}
                for tag, value in difference(ranks).items():
                    conditional[tag].append(float(value))
        if errors is not None:
            for tag, residual in errors.items():
                brier[tag].append(float(residual[idx].mean()))
    output = {}
    for key, rows in ap_auc.items():
        if len(rows) < BOOTSTRAPS * 0.95:
            raise ValueError("Too few valid hospital or patient bootstrap draws")
        bounds = np.quantile(rows, (0.025, 0.975), axis=0).T
        output[key] = {"ap": bounds[0].tolist(), "auroc": bounds[1].tolist(), "n": len(rows)}
        if within is not None:
            if len(conditional[key]) < BOOTSTRAPS * 0.95:
                raise ValueError("Too few hospital bootstrap draws contain within-hospital pairs")
            output[key]["within_hospital_auroc"] = np.quantile(conditional[key], (0.025, 0.975)).tolist()
    if errors is not None:
        output["calibration_brier"] = {key: np.quantile(v, (0.025, 0.975)).tolist() for key, v in brier.items()}
    return output


def leave_block_out(y, scores, groups):
    results = {tag: [] for tag in PAIRS}
    for held in groups:
        selected = np.ones(len(y), dtype=bool)
        selected[held] = False
        ranks = {tag: ranking(y[selected], scores[tag][selected])[0] for tag in MODELS}
        for tag, value in difference(ranks).items():
            results[tag].append(float(value))
    return {
        tag: {"min_ap_delta": min(values), "max_ap_delta": max(values),
              "positive_ap_delta": int(np.count_nonzero(np.asarray(values) > 0)), "n": len(values)}
        for tag, values in results.items()
    }


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    path42 = P42 / "summary.json"
    path28, path38 = P28 / "summary.json", ABLATION / "evaluation/summary.json"
    calibration, p28, p38 = map(from_json, (path42, path28, path38))
    protected = {
        **calibration["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in (path42, path28, path38)},
    }
    verify_sources(protected)
    columns = pd.read_parquet(AB_FEATURES).columns.tolist()
    with tempfile.TemporaryDirectory(dir=OUT, prefix="hospital-variation-") as temp:
        stage = Path(temp)
        probabilities, cohort, hashes = demo_scores(stage, columns, protected)
        if hashes != calibration["raw_eicu_sha256"] or hashes != p38["eicu_raw_sha256"]:
            raise ValueError("The original Demo sources differ from P38/P42")
        sites, blocks, links, patients, metadata = hospital_blocks(cohort)
        y = cohort.y.to_numpy()
        if (len(y), int(y.sum()), metadata["patients"]) != (782, 91, 719):
            raise ValueError("Demo cohort differs from previous comparisons")
        p28_reference = {(r["alignment"], r["mode"]): r for r in p28["eicu"]}
        p38_reference = {(r["alignment"], r["mode"]): r for r in p38["eicu"]}
        results = {}
        for case, scores in probabilities.items():
            alignment, mode = case.split(":")
            observed = p38_reference[alignment, mode]
            denominators, contributions, hospitals = within_hospital(y, scores, sites, links, len(blocks))
            errors = {
                tag: (calibrate(scores[tag], calibration["recipes"][tag]) - y) ** 2 - (scores[tag] - y) ** 2
                for tag in CALIBRATED
            }
            for tag, residual in errors.items():
                old = calibration["eicu"][case][tag]
                np.testing.assert_allclose(
                    residual.mean(), old["calibrated"]["brier"] - old["raw"]["brier"], rtol=0, atol=1e-12,
                )
            patient_ci = bootstrap(y, scores, patients)
            for tag, expected in (
                ("p38_vs_p28", observed["paired_ci_vs_p28"]),
                ("p38_vs_frozen_elastic", observed["paired_ci_vs_frozen_elastic"]),
            ):
                if patient_ci[tag] != expected:
                    raise ValueError(f"Original patient bootstrap differs in {case} for {tag}")
            if alignment == "urine_mapped":
                expected = p28_reference[alignment, mode]["comparison"]["frozen_elastic_mapped"]["paired_ci"]
                if patient_ci["p28_vs_frozen_elastic"] != expected:
                    raise ValueError(f"Original P28 patient bootstrap differs in {case}")
            block_ci = bootstrap(y, scores, blocks, within=(denominators, contributions), errors=errors)
            within_auc = {tag: float(row.sum() / denominators.sum()) for tag, row in contributions.items()}
            results[case] = {
                "pooled_ap_auroc": {tag: ranking(y, scores[tag]) for tag in MODELS},
                "within_hospital_auroc": within_auc,
                "within_hospital_auc_delta": {key: float(value) for key, value in difference(within_auc).items()},
                "within_hospital_pairs": int(denominators.sum()), "hospitals_with_pairs": hospitals,
                "patient_ci": patient_ci, "hospital_block_ci": block_ci,
                "omit_hospital_block": leave_block_out(y, scores, blocks),
                "calibration_brier_delta": {tag: float(values.mean()) for tag, values in errors.items()},
            }
            print(f"P43 {case}: verified patient intervals and completed hospital checks", flush=True)
        if source_hashes(DEMO) != hashes:
            raise ValueError("Demo data changed during the hospital audit")
        verify_sources(protected)
        output = stage / OUTPUT.name
        output.mkdir()
        save_json(output / "summary.json", {
            "device": "cpu", "fits": 0, "stays": len(y), "deaths": int(y.sum()),
            "cohort": metadata, "bootstraps": BOOTSTRAPS, "seed": SEED, "cases": results,
            "raw_eicu_sha256": hashes, "source_sha256": protected,
        })
        output.rename(OUTPUT)
    print(f"P43 hospital audit saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    run()
