"""Evaluate a frozen P33 sequence mixture without fitting on either test cohort."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from prepare_hourly_sequence import GROUPS, ID, MINUTE, PREPARED, ROOT, VALUE, hidden_channels, hourly_cube, static_values

import numpy as np
import pandas as pd

from eicu_demo_features import DEMO, STAY
from evaluate_eicu_confirmation import frozen_ensemble, frozen_models, mapped_inputs
from evaluate_eicu_demo import source_hashes
from evaluate_eicu_urine_mapping import urine_events
from evaluate_gcs_routed_temporal import RoutedPredictor, checked_metrics
from evaluate_ma_gbt_p12 import P11, REFERENCES, plan_for
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import benchmarks, compare_c, no_category_warning
from evaluate_robust_ensemble import EVALUATION as P28_EVAL, Ensemble
from evaluate_temporal_features import eicu_temporal, save_json, set_c_temporal
from pilot_temporal_features import time_summary
from prepare import OUT, sha
from stress_missingness_c import METRICS, SCENARIOS, apply_masks, load_cohort, mask_draws, score_metrics
from train_predictive_boosting import AB_FEATURES, from_json, verify_sources

CADENCE = ROOT.parent / "p34_sequence_cadence"
NEW = "sequence_mixture"
P28 = "p28"
WORKER = Path(__file__).with_name("infer_hourly_sequence.py")
PYTORCH = OUT / "ma_gbt_runtime" / ".venv" / "Scripts" / "python.exe"


def infer(stage, model, *, four_hour=False, **arrays):
    packet, result = stage / "packet.npz", stage / "predictions.npy"
    np.savez_compressed(packet, **arrays)
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, (ROOT / "cpu_deps", WORKER.parent))),
    }
    command = [str(PYTORCH), str(WORKER), str(model), str(packet), str(result)]
    if four_hour:
        command.append("--four-hour")
    subprocess.run(command, check=True, env=env)
    scores = np.load(result, allow_pickle=False)
    if not (np.isfinite(scores) & (scores >= 0) & (scores <= 1)).all():
        raise ValueError("Invalid frozen CNN output")
    return scores


def mask_key(hidden):
    return np.packbits(hidden.reshape(-1)).tobytes()


class SetCSequenceMixture:
    def __init__(self, original, anchor, seq_scores, weight):
        self.original, self.anchor, self.seq_scores, self.weight = original, anchor, seq_scores, weight

    def predict_proba(self, frame):
        if not self.original.index.equals(frame.index):
            raise ValueError("Unexpected row order in Set C")
        key = mask_key(hidden_channels(self.original, frame))
        score = self.weight * self.seq_scores[key].astype(float)
        if self.weight < 1.0:
            score += (1 - self.weight) * self.anchor.predict_proba(frame)[:, 1]
        return np.column_stack((1 - score, score))


def set_c(stage, model, weight, frozen_routed, gcs, rule, prior, reference, p11, cadence, p33_control):
    raw, y, _ = load_cohort()
    positions, plan, mask_digest = plan_for(raw)
    if mask_digest != reference["mask_draws_sha256"]:
        raise ValueError("Set C mask draw positions changed")
    timing, cube = set_c_temporal(raw.index, extension=lambda records, index: (
        time_summary(records, index), hourly_cube(records, index)
    ))
    no_mask = np.zeros(cube.shape[:2], dtype=bool)
    all_cases = {mask_key(no_mask): no_mask}
    draws = mask_draws(positions, plan)
    for _, variables, design in SCENARIOS:
        for draw in draws if design == "matched" else [{}]:
            masked = apply_masks(raw, {key: draw.get(key, positions[key]) for key in variables}, plan, design)
            hidden = hidden_channels(raw, masked)
            all_cases.setdefault(mask_key(hidden), hidden)
    static = static_values(raw)
    values = infer(stage, model, four_hour=cadence, cubes=cube, statics=static, hidden=np.stack(list(all_cases.values())))
    if values.shape != (len(all_cases), len(raw)):
        raise ValueError("Invalid Set C sequence score shape")
    seq_scores = dict(zip(all_cases, values, strict=True))
    anchor = Ensemble(RoutedPredictor(frozen_routed.rule, frozen_routed.models, raw, timing), gcs, rule)
    baseline = {row["scenario"]: row for row in prior["set_c"]}
    checked_metrics(y, anchor.predict_proba(raw)[:, 1], baseline["clean"])
    cols = raw.columns.tolist()
    predictor = SetCSequenceMixture(raw, anchor, seq_scores, weight)
    controls = {**benchmarks(), P28: (anchor, cols)}
    if p33_control:
        p33_path, p33_prior = p33_control
        p33_clean, = infer(stage, p33_path, cubes=cube[None], statics=static[None])
        p33_score = SetCSequenceMixture(raw, anchor, {mask_key(no_mask): p33_clean}, 0.25)
        checked_metrics(y, p33_score.predict_proba(raw)[:, 1], p33_prior["set_c"][0])
        controls["p33"] = (p33_score, cols)
    print(f"Verified {len(all_cases)} distinct Set C sequence masks", flush=True)
    summary, masks, intervals = compare_c({NEW: (predictor, cols)}, controls, reference, p11)
    for row in summary:
        for metric in METRICS:
            row[f"{metric}_p28"] = baseline[row["scenario"]][metric]
            row[f"delta_{metric}_vs_p28"] = row[metric] - row[f"{metric}_p28"]
    if p33_control:
        p33_reference = {row["scenario"]: row for row in p33_control[1]["set_c"]}
        for row in summary:
            row["ap_p33"] = p33_reference[row["scenario"]]["ap"]
            row["delta_ap_vs_p33"] = row["ap"] - row["ap_p33"]
    return summary, masks, intervals


def external(stage, model, weight, anchor, elastic, prior, columns, extras, cadence, p33_control):
    cubes = []

    def extension(records, index):
        cubes.append(hourly_cube(records, index))
        return time_summary(records, index)

    (cohort, raw), raw_hashes = eicu_temporal(extension=extension)
    if raw_hashes != prior["eicu_raw_sha256"]:
        raise ValueError("Demo sources differ from P28")
    y = cohort.y.to_numpy()
    urine_events_frame, _ = urine_events(cohort.index)
    urine_records = urine_events_frame.rename(columns={STAY: ID, "minute": MINUTE, "value": VALUE})
    urine = hourly_cube(urine_records, cohort.index)[:, GROUPS.index("Urine")]
    mapped = {mode: frame.copy() for mode, frame in raw.items()}
    mapped_inputs(DEMO, cohort, mapped)
    cases = []
    for alignment, frames in (("original", raw), ("urine_mapped", mapped)):
        for (mode, frame), cube in zip(frames.items(), cubes, strict=True):
            if alignment == "urine_mapped":
                cube = cube.copy()
                cube[:, GROUPS.index("Urine")] = urine
            cases.append((alignment, mode, frame.loc[:, columns + extras], cube))
    batch = {
        "cubes": np.stack([cube for _, _, _, cube in cases]),
        "statics": np.stack([static_values(frame) for _, _, frame, _ in cases]),
    }
    seq = infer(stage, model, four_hour=cadence, **batch)
    p33_scores, p33_old = {}, {}
    if p33_control:
        p33_path, p33_prior = p33_control
        p33_seq = infer(stage, p33_path, **batch)
        p33_old = {(row["alignment"], row["mode"]): row["metrics"] for row in p33_prior["eicu"]}
    if seq.shape != (len(cases), len(y)):
        raise ValueError("Invalid eICU sequence score shape")
    old = {(r["alignment"], r["mode"]): r for r in prior["eicu"]}
    rows = []
    first, previous = {}, {}
    frozen_comparison, frozen_reference = {}, {}
    pipe, cols = elastic
    for i, (alignment, mode, frame, _) in enumerate(cases):
        key = f"{alignment}:{mode}"
        p28_score = anchor.predict_proba(frame)[:, 1]
        checked_metrics(y, p28_score, old[alignment, mode]["metrics"])
        scores = weight * seq[i].astype(float) + (1 - weight) * p28_score
        first[key], previous[key] = scores, p28_score
        control = no_category_warning(pipe, frame.loc[:, cols])
        if alignment == "urine_mapped":
            checked_metrics(y, control, old[alignment, mode]["comparison"]["frozen_elastic_mapped"]["metrics"])
        frozen_comparison[key], frozen_reference[key] = scores, control
        if p33_control:
            p33_mix = 0.25 * p33_seq[i].astype(float) + 0.75 * p28_score
            checked_metrics(y, p33_mix, p33_old[alignment, mode])
            p33_scores[key] = p33_mix
        rows.append({
            "alignment": alignment, "mode": mode, "key": key,
            "metrics": score_metrics(y, scores), "sequence_only": score_metrics(y, seq[i]),
            "p28": score_metrics(y, p28_score), "frozen_elastic": score_metrics(y, control),
        })
    versus_p28 = paired_intervals(cohort, previous, first)
    versus_elastic = paired_intervals(cohort, frozen_reference, frozen_comparison)
    if p33_control:
        versus_p33 = paired_intervals(cohort, p33_scores, first)
    for row in rows:
        key = row.pop("key")
        row["paired_ci_vs_p28"] = versus_p28[key]
        row["paired_ci_vs_frozen_elastic"] = versus_elastic[key]
        if p33_control:
            row["p33"] = score_metrics(y, p33_scores[key])
            row["paired_ci_vs_p33"] = versus_p33[key]
    if source_hashes(DEMO) != raw_hashes:
        raise ValueError("Demo sources changed during scoring")
    return rows, raw_hashes


def evaluate(cadence=False):
    root = CADENCE if cadence else ROOT
    output = root / "evaluation"
    if output.exists():
        raise FileExistsError(output)
    fit_path, input_path, model = root / "fitted/manifest.json", PREPARED / "manifest.json", root / "fitted/sequence.pt"
    fit, inputs = from_json(fit_path), from_json(input_path)
    if fit["selected"] is None:
        print("A/B retained the existing candidate. Its frozen holdout evaluation applies.", flush=True)
        return
    bundle_hashes = fit["input_sha256"] if cadence else {input_path.relative_to(OUT).as_posix(): fit["input_manifest_sha256"]}
    if fit["device"] != "cpu" or sha(model) != fit["model_sha256"] or (cadence and fit["block_hours"] != 4):
        raise ValueError("Unexpected frozen sequence model")
    verify_sources(bundle_hashes)
    routed, _, gcs, extras, protected = frozen_models()
    rule, pipe, cols = frozen_ensemble(protected)
    paths = [P28_EVAL / "summary.json", P11 / "summary.json", REFERENCES / "manifest.json"]
    prior, p11, reference = map(from_json, paths)
    protected.update(inputs["source_sha256"])
    protected.update(prior["source_sha256"])
    protected.update(bundle_hashes)
    protected.update({path.relative_to(OUT).as_posix(): sha(path) for path in (*paths, fit_path, input_path, model)})
    p33_control = None
    if cadence:
        p33_path, p33_eval = ROOT / "fitted/sequence.pt", ROOT / "evaluation/summary.json"
        p33_prior = from_json(p33_eval)
        if p33_prior["training"]["selected"] != ["plain", 8, 0.25]:
            raise ValueError("Unexpected P33 recipe")
        protected[p33_path.relative_to(OUT).as_posix()] = p33_prior["training"]["model_sha256"]
        protected[p33_eval.relative_to(OUT).as_posix()] = sha(p33_eval)
        p33_control = (p33_path, p33_prior)
    verify_sources(protected)
    _, _, weight = fit["selected"]
    columns = pd.read_parquet(AB_FEATURES).columns.tolist()
    with tempfile.TemporaryDirectory(dir=root, prefix="sequence-evaluation-") as temporary:
        stage = Path(temporary)
        c, masks, c_ci = set_c(stage, model, weight, routed, gcs, rule, prior, reference, p11, cadence, p33_control)
        anchor = Ensemble(routed, gcs, rule)
        eicu, raw_sources = external(stage, model, weight, anchor, (pipe, cols), prior, columns, extras, cadence, p33_control)
        verify_sources(protected)
        deliverable = stage / "evaluation"
        deliverable.mkdir()
        masks.to_csv(deliverable / "set_c_paired_replicates.csv", index=False)
        save_json(deliverable / "summary.json", {
            "training": fit, "set_c": c, "set_c_clean_ci": c_ci, "eicu": eicu,
            "source_sha256": protected, "eicu_raw_sha256": raw_sources,
        })
        deliverable.rename(output)
    print(f"Completed frozen CPU sequence evaluation at {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", action="store_true")
    evaluate(cadence=parser.parse_args().cadence)
