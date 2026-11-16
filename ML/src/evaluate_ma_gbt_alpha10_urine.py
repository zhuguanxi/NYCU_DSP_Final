"""Compare the urine augmentation with frozen alpha=10 and GCS-only models."""

import argparse
import json
import tempfile
import warnings
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE, encode

import numpy as np
import pandas as pd

from evaluate_eicu_demo import MODES, evaluate_predictions
from evaluate_ma_gbt_p12 import KEYS, METRICS, REFERENCES, eicu_inputs, plan_for
from evaluate_missingness_augmentation import augmented_models, paired_intervals
from prepare import OUT, sha
from stress_missingness_c import apply_masks, load_cohort, score_metrics, stress
from train_missingness_augmentation import GCS_ONLY_OUTPUT, MODELS, from_json

P15 = OUT / "ma_gbt_alpha10"
OUTPUT = OUT / "ma_gbt_alpha10_urine"
P15_EVALUATION = P15 / "evaluation"
GCS_REFERENCE = OUTPUT / "gcs_urine_reference.json"
EVALUATION = OUTPUT / "evaluation"
BASELINES = ("ma10", "aug10", "gcs")
NEW = "urine10"
RHO_SCENARIOS = ("clean", "full_Urine")


def c_inputs(reference):
    frame, y, _ = load_cohort()
    positions, plan, digest = plan_for(frame)
    if digest != reference["mask_draws_sha256"]:
        raise ValueError("P9 matched-mask positions differ from the frozen comparisons")
    views = {"clean": frame, "full_Urine": apply_masks(frame, {"Urine": positions["Urine"]}, plan, "full")}
    return frame, y, positions, plan, views


def xgb_missing_path_hits(pipeline, frame, schema, selected=None):
    raw = frame.loc[:, schema["columns"]]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        values = pipeline.named_steps["pre"].transform(raw)
    expected, missing = encode(raw, schema, dtype=np.float64)
    np.testing.assert_array_equal(values, expected)
    model = pipeline.named_steps["model"]
    booster = model.get_booster()
    feature_names = booster.feature_names or [f"f{i}" for i in range(values.shape[1])]
    indices = {
        name: i for i, name in enumerate(feature_names) if selected is None or i in selected
    }
    # XGBoost supplies the actual visited terminal node in each tree.
    terminal = np.asarray(model.apply(values), dtype=np.int64).reshape(len(raw), -1)
    trees = booster.get_dump(dump_format="json")
    if len(trees) < terminal.shape[1]:
        raise ValueError("XGBoost returned more leaf columns than there are trees")
    relied = np.zeros_like(missing, dtype=bool)
    for j in range(terminal.shape[1]):
        unmatched = set(np.unique(terminal[:, j]))
        pending = [(json.loads(trees[j]), ())]
        while pending:
            node, path = pending.pop()
            if "leaf" in node:
                unmatched.discard(node["nodeid"])
                if path:
                    rows = np.flatnonzero(terminal[:, j] == node["nodeid"])
                    cells = np.ix_(rows, path)
                    relied[cells] |= missing[cells].astype(bool)
                continue
            if node["split"] in indices:
                path = (*path, indices[node["split"]])
            pending.extend((child, path) for child in node["children"])
        if unmatched:
            raise ValueError("An XGBoost terminal node is absent from the corresponding tree")
    probabilities = model.predict_proba(values)[:, 1]
    return relied, probabilities


def xgb_urine_rho(pipeline, frame, schema):
    selected = {i for i, name in enumerate(schema["numeric"]) if name.startswith("Urine__")}
    relied, probabilities = xgb_missing_path_hits(pipeline, frame, schema, selected)
    return float(relied.any(axis=1).mean()), probabilities


def export_gcs_reference():
    if GCS_REFERENCE.exists():
        raise FileExistsError(GCS_REFERENCE)
    frozen = from_json(OUTPUT / "manifest.json")
    if (frozen["training_dataset"], frozen["device"]) != ("P12 A/B", "cpu"):
        raise ValueError("Expected frozen P16 CPU models before holdout access")
    reference_path = REFERENCES / "manifest.json"
    reference = from_json(reference_path)
    prior_path = P15_EVALUATION / "summary.json"
    prior = from_json(prior_path)
    schemas = from_json(SCHEMA_FILE)["schemas"]
    paths, pipelines, manifest = augmented_models(GCS_ONLY_OUTPUT)
    if set(manifest["selected_rates_percent"]) != {"GCS"}:
        raise ValueError("Expected GCS-only augmentation")
    _, y, _, _, c_views = c_inputs(reference)
    cohort, eicu = eicu_inputs()
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    if sha(gcs_path) != reference["gcs_scores_sha256"]:
        raise ValueError("Stored GCS-only eICU predictions differ")
    gcs_scores = pd.read_parquet(gcs_path)
    if not gcs_scores.index.equals(cohort.index):
        raise ValueError("Stored GCS-only eICU predictions do not align")
    c_prior = {(row["scenario"], row["model"]): row for row in prior["set_c"]}
    c_rho, eicu_rho = {}, {}
    for name, (pipeline, columns) in pipelines.items():
        schema = schemas[name]["final"]
        if list(columns) != schema["columns"]:
            raise ValueError("GCS-only columns differ from the frozen schema")
        for scenario, frame in c_views.items():
            rho, score = xgb_urine_rho(pipeline, frame, schema)
            measures = score_metrics(y, score)
            expected = c_prior[scenario, name]
            np.testing.assert_allclose(
                [measures[m] for m in METRICS], [expected[f"{m}_gcs"] for m in METRICS], atol=1e-12, rtol=0
            )
            c_rho[f"{scenario}_{name}"] = rho
        for mode in MODES:
            key = f"{mode}_{name}"
            eicu_rho[key], score = xgb_urine_rho(pipeline, eicu[mode], schema)
            np.testing.assert_array_equal(score, gcs_scores[key].to_numpy(dtype=score.dtype))
    sources = {
        **{path.relative_to(OUT).as_posix(): sha(path) for path in (
            reference_path, prior_path, SCHEMA_FILE, gcs_path, *paths.values()
        )},
        **{path: digest for path, digest in reference["source_sha256"].items()},
    }
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="gcs-urine-") as temporary:
        staging = Path(temporary) / GCS_REFERENCE.name
        staging.write_text(
            json.dumps({"set_c": c_rho, "eicu": eicu_rho, "source_sha256": sources}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(GCS_REFERENCE)
    print("Verified the frozen GCS-only scores and measured urinary reliance from native leaf paths", flush=True)


def ma_urine_rho(pipe, frame):
    from train_ma_gbt_alpha10_urine import urine_mask

    features, mask = encode(frame, pipe.schema)
    return float(pipe.model.compute_missingness_reliance(features, urine_mask(mask, pipe.schema)))


def compare_set_c(pipes, reference, prior, gcs):
    frame, y, positions, plan, views = c_inputs(reference)
    new = pipes[NEW]
    scores = {
        f"score_{name}": pipe.predict_proba(frame.loc[:, columns])[:, 1] for name, (pipe, columns) in new.items()
    }
    baseline, replicates = stress(frame, y, pd.DataFrame(scores, index=frame.index), new, positions, plan)
    old = pd.read_csv(P15_EVALUATION / "set_c_paired_replicates.csv")
    columns = [f"{metric}_{tag}" for tag in BASELINES for metric in METRICS]
    paired = old[KEYS + columns].merge(
        replicates[KEYS + list(METRICS)].rename(columns={metric: f"{metric}_{NEW}" for metric in METRICS}),
        on=KEYS, validate="one_to_one",
    )
    if not len(paired) == len(old) == len(replicates) == 336:
        raise ValueError("Set C did not reproduce every P15 mask replicate")
    for tag in BASELINES:
        for metric in METRICS:
            paired[f"delta_{metric}_vs_{tag}"] = paired[f"{metric}_{NEW}"] - paired[f"{metric}_{tag}"]
    groups = {key: rows for key, rows in paired.groupby(["scenario", "model"], sort=False)}
    rows = []
    for previous in prior["set_c"]:
        scenario, name = previous["scenario"], previous["model"]
        row = {**{key: previous[key] for key in ("scenario", "model", "repeats")}, **{col: previous[col] for col in columns}}
        group = groups.get((scenario, name))
        if group is not None:
            if len(group) != row["repeats"]:
                raise ValueError("Set C scenario repeat count changed")
            np.testing.assert_allclose(group[columns].mean(), [row[col] for col in columns], rtol=0, atol=1e-12)
        elif scenario != "clean":
            raise ValueError("Set C was missing a masked scenario")
        for metric in METRICS:
            current = baseline[name][metric] if group is None else float(group[f"{metric}_{NEW}"].mean())
            row[f"{metric}_{NEW}"] = current
            for tag in BASELINES:
                col = f"delta_{metric}_vs_{tag}"
                row[col] = current - row[f"{metric}_{tag}"]
                if group is not None:
                    row[f"{col}_p10_p90"] = group[col].quantile([0.1, 0.9]).tolist()
        if scenario in RHO_SCENARIOS:
            row["urine_rho_gcs"] = gcs["set_c"][f"{scenario}_{name}"]
            for tag, model_group in pipes.items():
                pipe, features = model_group[name]
                row[f"urine_rho_{tag}"] = ma_urine_rho(pipe, views[scenario].loc[:, features])
        rows.append(row)
    return rows, paired


def compare_eicu(pipes, reference, prior, gcs):
    from evaluate_ma_gbt_augmentation import eicu_scores, reference_metrics

    cohort, frames = eicu_inputs()
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    if sha(gcs_path) != reference["gcs_scores_sha256"]:
        raise ValueError("Frozen GCS-only eICU scores changed")
    old_gcs = pd.read_parquet(gcs_path)
    if not old_gcs.index.equals(cohort.index):
        raise ValueError("GCS-only eICU scores do not align")
    scores, rhos = {}, {}
    for tag, group in pipes.items():
        scores[tag], rhos[tag] = eicu_scores(group, frames)
    scores["gcs"] = {key: old_gcs[key].to_numpy() for key in scores[NEW]}
    targets = cohort.y.to_numpy()
    history = {(row["mode"], row["model"]): row for row in prior["eicu"]}
    for (mode, name), row in history.items():
        key = f"{mode}_{name}"
        for tag in BASELINES:
            metrics = reference_metrics(targets, scores[tag][key])
            np.testing.assert_allclose(
                [metrics[m] for m in METRICS], [row[f"{m}_{tag}"] for m in METRICS], atol=1e-12, rtol=0
            )
            if tag in rhos:
                np.testing.assert_allclose(rhos[tag][key], row[f"rho_{tag}"], rtol=0, atol=1e-12)
    print("Evaluating both original eICU cadences and paired patient-level differences", flush=True)
    absolute = evaluate_predictions(cohort, scores[NEW])
    contrasts = {tag: paired_intervals(cohort, scores[tag], scores[NEW]) for tag in BASELINES}
    rows = []
    for measures in absolute:
        mode, name = measures["mode"], measures["model"]
        key = f"{mode}_{name}"
        old = history[mode, name]
        row = {"mode": mode, "model": name, "new_metrics": measures, "urine_rho_gcs": gcs["eicu"][key]}
        for tag, group in pipes.items():
            pipe, columns = group[name]
            row[f"urine_rho_{tag}"] = ma_urine_rho(pipe, frames[mode].loc[:, columns])
            row[f"rho_{tag}"] = rhos[tag][key]
        for metric in METRICS:
            row[f"{metric}_{NEW}"] = measures[metric]
            for tag in BASELINES:
                row[f"{metric}_{tag}"] = old[f"{metric}_{tag}"]
                row[f"delta_{metric}_vs_{tag}"] = measures[metric] - old[f"{metric}_{tag}"]
                if metric in contrasts[tag][key]:
                    row[f"delta_{metric}_vs_{tag}_ci"] = contrasts[tag][key][metric]
        rows.append(row)
    return rows


def evaluate():
    from evaluate_ma_gbt_alpha10 import split_counts
    from evaluate_ma_gbt_augmentation import pipelines
    from run_ma_gbt_p12 import validate_upstream
    from train_ma_gbt_alpha10 import ALPHA, RATES, verify_hashes
    from train_ma_gbt_alpha10_urine import URINE_CANDIDATES, sources_for_p15

    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    validate_upstream()
    fit_path = OUTPUT / "manifest.json"
    fit = from_json(fit_path)
    prior_fit, source = sources_for_p15()
    if (
        fit["source_sha256"] != source or fit["alpha"] != ALPHA or fit["fixed_rates_percent"] != RATES
        or fit["selected_urine_rate_percent"] not in URINE_CANDIDATES
        or (fit["training_dataset"], fit["device"]) != ("P12 A/B", "cpu")
    ):
        raise ValueError("Expected the frozen P16 A/B training manifest")
    prior_path = P15_EVALUATION / "summary.json"
    prior = from_json(prior_path)
    if prior["training"] != prior_fit:
        raise ValueError("P15 metrics did not use the expected frozen models")
    reference_path = REFERENCES / "manifest.json"
    reference = from_json(reference_path)
    gcs = from_json(GCS_REFERENCE)
    extra = (fit_path, prior_path, reference_path, GCS_REFERENCE, P15_EVALUATION / "set_c_paired_replicates.csv")
    protected = {
        **source, **prior["source_sha256"], **reference["source_sha256"], **gcs["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in extra},
        **{(OUTPUT / f"{name}.joblib").relative_to(OUT).as_posix(): fit["model_sha256"][name] for name in MODELS},
    }
    verify_hashes(protected)
    schemas = from_json(SCHEMA_FILE)["schemas"]
    pipes = {tag: pipelines(P15 / tag, schemas) for tag in ("ma10", "aug10")}
    pipes[NEW] = pipelines(OUTPUT, schemas)
    set_c, masked = compare_set_c(pipes, reference, prior, gcs)
    eicu = compare_eicu(pipes, reference, prior, gcs)
    verify_hashes(protected)
    report = {
        "training": fit, "set_c": set_c, "eicu": eicu,
        "source_sha256": protected, "split_counts": split_counts(pipes),
    }
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="ma-gbt-urine-eval-") as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        masked.to_csv(staging / "set_c_paired_replicates.csv", index=False)
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(EVALUATION)
    print(f"Urine augmentation comparison saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-reference", action="store_true", help="Run in the original XGBoost environment")
    if parser.parse_args().gcs_reference:
        export_gcs_reference()
    else:
        evaluate()
