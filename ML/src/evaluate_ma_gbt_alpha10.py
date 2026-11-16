"""Compare the four frozen MA-GBT arms and GCS-only on the fixed holdouts."""

import json
import tempfile
from collections import Counter
from pathlib import Path

from ma_gbt_p12_inputs import SCHEMA_FILE

import numpy as np
import pandas as pd

from evaluate_eicu_demo import MODES, evaluate_predictions
from evaluate_ma_gbt_augmentation import EVALUATION as P14_EVALUATION
from evaluate_ma_gbt_augmentation import eicu_scores, pipelines, reference_metrics
from evaluate_ma_gbt_p12 import FITTED as P13_FITTED
from evaluate_ma_gbt_p12 import KEYS, METRICS, REFERENCES, check_fit, eicu_inputs, plan_for
from evaluate_missingness_augmentation import paired_intervals
from prepare import OUT, sha
from run_ma_gbt_p12 import validate_upstream
from stress_missingness_c import load_cohort, stress
from train_ma_gbt_alpha10 import ALPHA, OUTPUT, RATES, RECIPES_BY_RUN, inputs, verify_hashes
from train_ma_gbt_augmentation import OUTPUT as P14_FITTED
from train_missingness_augmentation import MODELS, from_json

EVALUATION = OUTPUT / "evaluation"
OLD_TAGS = {"ma": "ma1000", "aug": "aug1000", "gcs": "gcs"}
TAGS = (*RECIPES_BY_RUN, *OLD_TAGS.values())
CONTRASTS = {
    "ma10_vs_ma1000": ("ma10", "ma1000"),
    "aug10_vs_aug1000": ("aug10", "aug1000"),
    "aug10_vs_ma10": ("aug10", "ma10"),
    "ma10_vs_gcs": ("ma10", "gcs"),
    "aug10_vs_gcs": ("aug10", "gcs"),
}


def all_metrics(tags):
    return [f"{metric}_{tag}" for tag in tags for metric in METRICS]


def old_metrics(values):
    return {
        f"{metric}_{tag}": values[f"{metric}_{old}"]
        for old, tag in OLD_TAGS.items() for metric in METRICS
    }


def add_contrasts(values):
    for contrast, (after, before) in CONTRASTS.items():
        for metric in METRICS:
            values[f"delta_{metric}_{contrast}"] = values[f"{metric}_{after}"] - values[f"{metric}_{before}"]


def compare_set_c(new_pipes, reference, prior):
    frame, y, _ = load_cohort()
    positions, plan, digest = plan_for(frame)
    if digest != reference["mask_draws_sha256"]:
        raise ValueError("Set C picked different matched-mask positions from P13/P14")
    flattened = {f"{tag}_{name}": entry for tag, pipes in new_pipes.items() for name, entry in pipes.items()}
    scores, rhos = {}, {}
    for key, (pipe, columns) in flattened.items():
        scores[f"score_{key}"], rhos[key] = pipe.predict_and_rho(frame.loc[:, columns])
    baseline, new = stress(frame, y, pd.DataFrame(scores, index=frame.index), flattened, positions, plan)
    new[["variant", "model"]] = new.model.str.split("_", n=1, expand=True)
    previous = pd.read_csv(P14_EVALUATION / "set_c_paired_replicates.csv")
    changes = {
        f"{metric}_{old}": f"{metric}_{tag}"
        for old, tag in OLD_TAGS.items() for metric in METRICS
    }
    paired = previous.rename(columns=changes)[KEYS + list(changes.values())]
    for tag in RECIPES_BY_RUN:
        renamed = {metric: f"{metric}_{tag}" for metric in METRICS}
        current = new.loc[new.variant.eq(tag), KEYS + list(METRICS)].rename(columns=renamed)
        paired = paired.merge(current, on=KEYS, validate="one_to_one")
        if not len(current) == len(previous) == len(paired) == 336:
            raise ValueError(f"{tag} did not reproduce every P14 mask replicate")
    add_contrasts(paired)
    groups = {key: data for key, data in paired.groupby(["scenario", "model"], sort=False)}
    clean = {row["model"]: old_metrics(row) for row in prior["set_c"] if row["scenario"] == "clean"}
    for name in MODELS:
        for tag in RECIPES_BY_RUN:
            clean[name].update({f"{metric}_{tag}": baseline[f"{tag}_{name}"][metric] for metric in METRICS})

    rows = []
    for old in prior["set_c"]:
        scenario, name = old["scenario"], old["model"]
        values = {"scenario": scenario, "model": name, "repeats": old["repeats"]}
        if scenario == "clean":
            values.update(clean[name])
            for tag in RECIPES_BY_RUN:
                values[f"rho_{tag}"] = rhos[f"{tag}_{name}"]
            for historical in ("ma", "aug"):
                values[f"rho_{OLD_TAGS[historical]}"] = old[f"rho_{historical}"]
        else:
            matched = groups[scenario, name]
            if len(matched) != old["repeats"]:
                raise ValueError("Set C scenario replicate count changed")
            values.update(matched[all_metrics(TAGS)].mean().to_dict())
            for metric, expected in old_metrics(old).items():
                np.testing.assert_allclose(values[metric], expected, rtol=0, atol=1e-12)
            for tag in TAGS:
                values[f"delta_ap_from_clean_{tag}"] = values[f"ap_{tag}"] - clean[name][f"ap_{tag}"]
            for contrast in CONTRASTS:
                for metric in METRICS:
                    col = f"delta_{metric}_{contrast}"
                    values[f"{col}_p10_p90"] = matched[col].quantile([0.1, 0.9]).tolist()
        add_contrasts(values)
        rows.append(values)
    return rows, paired


def compare_eicu(pipes, reference, prior):
    cohort, frames = eicu_inputs()
    y = cohort.y.to_numpy()
    scores, rhos = {}, {}
    for tag, group in pipes.items():
        scores[tag], rhos[tag] = eicu_scores(group, frames)
    gcs_path = REFERENCES / "gcs_eicu_scores.parquet"
    if sha(gcs_path) != reference["gcs_scores_sha256"]:
        raise ValueError("Frozen GCS-only eICU predictions changed")
    gcs = pd.read_parquet(gcs_path)
    if not gcs.index.equals(cohort.index):
        raise ValueError("GCS-only eICU predictions do not align to the original cohort")
    keys = [f"{mode}_{name}" for mode in MODES for name in MODELS]
    scores["gcs"] = {key: gcs[key].to_numpy() for key in keys}
    for row in prior["eicu"]:
        key = f"{row['mode']}_{row['model']}"
        for old, tag in OLD_TAGS.items():
            observed = reference_metrics(y, scores[tag][key])
            np.testing.assert_allclose(
                [observed[m] for m in METRICS], [row[f"{m}_{old}"] for m in METRICS], rtol=0, atol=1e-12
            )
            if tag in rhos:
                np.testing.assert_allclose(rhos[tag][key], row[f"rho_{old}"], rtol=0, atol=1e-12)

    print("Evaluating the original eICU cadences and patient-paired differences", flush=True)
    absolute = {
        tag: {(row["mode"], row["model"]): row for row in evaluate_predictions(cohort, scores[tag])}
        for tag in RECIPES_BY_RUN
    }
    intervals = {
        contrast: paired_intervals(cohort, scores[before], scores[after])
        for contrast, (after, before) in CONTRASTS.items()
    }
    output = []
    for elapsed in prior["eicu"]:
        mode, name = elapsed["mode"], elapsed["model"]
        key = f"{mode}_{name}"
        row = {"mode": mode, "model": name, **old_metrics(elapsed)}
        for tag in RECIPES_BY_RUN:
            measures = absolute[tag][mode, name]
            row.update({f"{metric}_{tag}": measures[metric] for metric in METRICS})
            row[f"metrics_{tag}"] = measures
        for tag in pipes:
            row[f"rho_{tag}"] = rhos[tag][key]
        add_contrasts(row)
        for label, values in intervals.items():
            for metric in ("ap", "auroc"):
                row[f"delta_{metric}_{label}_ci"] = values[key][metric]
        output.append(row)
    return output


def split_counts(pipes):
    counts = {}
    for tag, group in pipes.items():
        counts[tag] = {}
        for name, (pipe, _) in group.items():
            schema = pipe.schema
            columns = schema["numeric"] + [
                cat["name"] for cat in schema["categorical"] for _ in cat["levels"]
            ]
            counts[tag][name] = dict(Counter(
                columns[index] for tree in pipe.model.estimators_.ravel()
                for index in tree.tree_.feature if index >= 0
            ))
    return counts


def evaluate():
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    validate_upstream()
    fit_path = OUTPUT / "manifest.json"
    fit = from_json(fit_path)
    if fit["alpha"] != ALPHA or fit["augmentation_rates_percent"] != RATES or fit["tuned_in_this_run"]:
        raise ValueError("Expected the prespecified alpha=10 and augmentation rates")
    if (fit["training_dataset"], fit["device"]) != ("P12 A/B", "cpu"):
        raise ValueError("Expected frozen CPU models fitted on A/B only")
    schemas, _, p14_fit, train_sources = inputs()
    if fit["source_sha256"] != train_sources:
        raise ValueError("Alpha=10 training provenance differs from the frozen sources")
    reference_path = REFERENCES / "manifest.json"
    reference = from_json(reference_path)
    prior_path = P14_EVALUATION / "summary.json"
    prior = from_json(prior_path)
    if prior["training"] != p14_fit:
        raise ValueError("P14 evaluation did not use the referenced models")
    extra = (fit_path, reference_path, prior_path, P14_EVALUATION / "set_c_paired_replicates.csv")
    model_digests = fit["model_sha256"]
    expected_paths = {f"{tag}/{name}.joblib" for tag in RECIPES_BY_RUN for name in MODELS}
    if set(model_digests) != expected_paths:
        raise ValueError("Expected exactly four frozen alpha=10 model artifacts")
    protected = {
        **train_sources, **reference["source_sha256"], **prior["source_sha256"],
        **{path.relative_to(OUT).as_posix(): sha(path) for path in extra},
        **{(OUTPUT / path).relative_to(OUT).as_posix(): digest for path, digest in model_digests.items()},
    }
    verify_hashes(protected)

    new = {tag: pipelines(OUTPUT / tag, schemas) for tag in RECIPES_BY_RUN}
    pipes = {**new, "ma1000": pipelines(P13_FITTED, schemas), "aug1000": pipelines(P14_FITTED, schemas)}
    set_c, replicates = compare_set_c(new, reference, prior)
    eicu = compare_eicu(pipes, reference, prior)
    verify_hashes(protected)
    check_fit()
    report = {
        "training": fit, "set_c": set_c, "eicu": eicu,
        "split_counts": split_counts(pipes), "source_sha256": protected,
    }
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="ma-gbt-alpha10-eval-") as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        replicates.to_csv(staging / "set_c_paired_replicates.csv", index=False)
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(EVALUATION)
    print(f"Alpha=10 comparison saved to {EVALUATION}", flush=True)


if __name__ == "__main__":
    evaluate()
