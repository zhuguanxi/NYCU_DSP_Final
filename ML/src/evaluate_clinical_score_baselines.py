"""Compare published clinical scores with frozen models on matched stay cohorts."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from audit_p12_outcome_timing import DEATH, OFFICIAL_SHA256, outcome_timing, replay_frozen_c
from eicu_demo_features import DEMO, MODES, STAY
from evaluate_eicu_demo import BOOTSTRAPS, SEED, ranking, source_hashes
from evaluate_frozen_calibration import demo_scores
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import C_BOOTSTRAPS, C_BOOTSTRAP_SEED, contrast_c
from evaluate_temporal_features import save_json
from prepare import DATA, OUT, sha
from stress_missingness_c import score_metrics
from train_predictive_boosting import ab_inputs, verify_sources

OUTPUT = OUT / "p48_clinical_score_baselines"
SCORE_COLUMNS = ("SAPS-I", "SOFA")
C_MODELS = ("frozen_full", "frozen_elastic", "p26", "p28", "p33", "p38")
DEMO_MODELS = ("frozen_full", "frozen_elastic", "p28", "p33", "p38")
APACHE_FILE = DEMO / "apachePatientResult.csv.gz"
APACHE_SHA256 = "ce799692737f7a712c59aa8fbb9a302cee7c719887f0638caed3ae144b166821"
RISK = "predictedhospitalmortality"


def counts(y, included):
    if set(y[included]) != {0, 1}:
        raise ValueError("A matched cohort must include both outcomes")
    return {
        "total_stays": len(y), "total_deaths": int(y.sum()),
        "matched_stays": int(included.sum()), "matched_deaths": int(y[included].sum()),
        "excluded_stays": int((~included).sum()), "excluded_deaths": int(y[~included].sum()),
    }


def development_scores(x, y):
    rows = []
    for split in ("a", "b"):
        path = DATA / f"Outcomes-{split}.txt"
        if sha(path) != OFFICIAL_SHA256[split]:
            raise ValueError(f"Unexpected P12 outcome table for {split}")
        rows.append(pd.read_csv(path, index_col="RecordID", usecols=["RecordID", *SCORE_COLUMNS, DEATH]))
    data = pd.concat(rows, verify_integrity=True).loc[x.index]
    np.testing.assert_array_equal(data[DEATH], y)
    common = data.loc[:, list(SCORE_COLUMNS)].ge(0).all(axis=1).to_numpy()
    models = {}
    summaries = {}
    for name in SCORE_COLUMNS:
        model = LogisticRegression(C=np.inf, tol=1e-10)
        model.fit(data.loc[common, [name]], y[common])
        slope = float(model.coef_[0, 0])
        if model.n_iter_[0] >= model.max_iter or not np.isfinite(slope) or slope <= 0:
            raise ValueError(f"Expected converged, increasing P12 risk map for {name}")
        models[name] = model
        summaries[name] = {"slope": slope, "intercept": float(model.intercept_[0])}
    return models, {"cohort": counts(y, common), "risk_maps": summaries}


def set_c_scores(stage, timing, models):
    raw, y, frozen, _, protected, mask_digest = replay_frozen_c(stage)
    if set(raw.index) != set(timing.index):
        raise ValueError("P12 outcome table and frozen Set C keys differ")
    clinical = timing.loc[raw.index, [*SCORE_COLUMNS, DEATH]]
    np.testing.assert_array_equal(clinical[DEATH], y)
    common = clinical.loc[:, list(SCORE_COLUMNS)].ge(0).all(axis=1).to_numpy()
    matched = clinical.loc[common, list(SCORE_COLUMNS)]
    cohort = counts(y, common)
    target = y[common]
    scores = {name: frozen["clean"][name][common] for name in C_MODELS}
    for name, model in models.items():
        pred = model.predict_proba(matched[[name]])[:, 1]
        np.testing.assert_allclose(ranking(target, pred), ranking(target, matched[name]), rtol=0, atol=1e-12)
        scores[name] = pred
    metrics = {name: score_metrics(target, pred) for name, pred in scores.items()}
    contrasts = contrast_c(
        target, {name: scores[name] for name in (*SCORE_COLUMNS, "frozen_full")}, {"p33": scores["p33"]},
    )
    return {
        "cohort": cohort, "invalid_score_counts": clinical.loc[:, list(SCORE_COLUMNS)].lt(0).sum().to_dict(),
        "metrics": metrics, "paired_ci": contrasts, "mask_sha256": mask_digest,
        "bootstrap_replicates": C_BOOTSTRAPS, "bootstrap_seed": C_BOOTSTRAP_SEED,
    }, protected


def demo_apache(cohort):
    if sha(APACHE_FILE) != APACHE_SHA256:
        raise ValueError("APACHE input does not match the official Demo 2.0.1 checksum")
    if cohort.index.name != STAY or not cohort.index.is_unique or cohort.uniquepid.isna().any():
        raise ValueError("Unexpected Demo stay or patient keys")
    apache = pd.read_csv(APACHE_FILE, usecols=[STAY, "apacheversion", RISK])
    iva = apache.loc[apache.apacheversion.eq("IVa"), [STAY, RISK]].set_index(STAY, verify_integrity=True)
    if len(iva) != 1838 or iva.index.hasnans:
        raise ValueError("Unexpected APACHE IVa release keys")
    matched = iva.reindex(cohort.index)
    risk = pd.to_numeric(matched[RISK], errors="coerce")
    included = risk.between(0, 1).to_numpy()
    available = cohort.index.isin(iva.index)
    summary = {
        **counts(cohort.y.to_numpy(), included),
        "matched_patients": int(cohort.loc[included, "uniquepid"].nunique()),
        "total_patients": int(cohort.uniquepid.nunique()),
        "no_iva_row_stays": int((~available).sum()), "invalid_risk_stays": int((available & ~included).sum()),
    }
    return risk.to_numpy()[included], included, summary


def demo_comparison(stage, columns, protected):
    frozen, cohort, hashes = demo_scores(stage, columns, protected)
    apache, included, summary = demo_apache(cohort)
    matched = cohort.loc[included]
    target = matched.y.to_numpy()
    scores = {
        mode: {name: frozen[f"urine_mapped:{mode}"][name][included] for name in DEMO_MODELS}
        for mode in MODES
    }
    comparisons = {f"{mode}:{name}": scores[mode][name] for mode in MODES for name in ("p28", "p38")}
    intervals = paired_intervals(matched, dict.fromkeys(comparisons, apache), comparisons)
    metrics = {
        mode: {name: score_metrics(target, pred) for name, pred in {"apache_iva": apache, **values}.items()}
        for mode, values in scores.items()
    }
    return {
        "cohort": summary, "metrics": metrics, "paired_ci": intervals, "alignment": "urine_mapped",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": SEED,
    }, hashes


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with tempfile.TemporaryDirectory(dir=OUT, prefix="p48-clinical-") as temporary, threadpool_limits(limits=2):
        stage = Path(temporary)
        timing, _, p12_hashes = outcome_timing()
        x, y, _ = ab_inputs()
        models, development = development_scores(x, y)
        print("Froze SAPS-I and SOFA probability maps from P12 A/B", flush=True)
        c, protected = set_c_scores(stage, timing, models)
        print("Scored matched Set C cases and paired intervals", flush=True)
        demo, demo_hashes = demo_comparison(stage, x.columns.tolist(), protected)
        verify_sources(protected)
        if any(sha(DATA / f"Outcomes-{split}.txt") != digest for split, digest in p12_hashes.items()):
            raise ValueError("A published P12 outcome source changed during evaluation")
        if source_hashes(DEMO) != demo_hashes or sha(APACHE_FILE) != APACHE_SHA256:
            raise ValueError("A published Demo source changed during evaluation")
        output = stage / OUTPUT.name
        output.mkdir()
        save_json(output / "summary.json", {
            "device": "cpu", "existing_prediction_model_fits": 0, "clinical_probability_fits": len(models),
            "development": development, "set_c": c, "eicu_demo": demo, "frozen_source_sha256": protected,
            "published_p12_sha256": p12_hashes, "raw_demo_sha256": {**demo_hashes, APACHE_FILE.name: APACHE_SHA256},
        })
        output.rename(OUTPUT)
    print(f"Saved aggregate CPU clinical baseline comparisons to {OUTPUT}", flush=True)


if __name__ == "__main__":
    evaluate()
