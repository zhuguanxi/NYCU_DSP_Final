"""Evaluate the frozen seed ensemble or audit the existing P38 versus P37 pair."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_eicu_confirmation import frozen_ensemble, frozen_models
from evaluate_hourly_sequence import (
    ABLATION, BLEND, CADENCE, GROUPS, PREPARED, SequenceReference, external, set_c,
)
from evaluate_ma_gbt_p12 import P11, REFERENCES
from evaluate_robust_ensemble import EVALUATION as P28_EVAL, Ensemble
from evaluate_temporal_features import save_json
from prepare import OUT, sha
from stress_missingness_c import METRICS
from train_predictive_boosting import AB_FEATURES, from_json, verify_sources

ROOT = OUT / "p39_sequence_seed_ensemble"


def channel_indices(included):
    if [group for group in GROUPS if group in included] != included:
        raise ValueError("The frozen channel order differs from the original input order")
    return [GROUPS.index(group) for group in included]


def evaluate():
    output = ROOT / "evaluation"
    if output.exists():
        raise FileExistsError(output)
    p39_path = ROOT / "fitted/manifest.json"
    p38_path, history38 = ABLATION / "fitted/manifest.json", ABLATION / "evaluation/summary.json"
    p39, p38, p38_eval = map(from_json, (p39_path, p38_path, history38))
    if (
        p38_eval["training"] != p38
        or (p39["dataset"], p39["device"], p39["block_hours"]) != ("P12 A/B", "cpu", 4)
    ):
        raise ValueError("Expected the frozen CPU sequence recipes")
    p38_model = p38_path.with_name("sequence.pt")
    p38_channels = channel_indices(p38["included_channels"])
    blend_rule = p39["selected"]
    routed, _, gcs, extras, protected = frozen_models()
    rule, elastic, reduced = frozen_ensemble(protected)
    input_path = PREPARED / "manifest.json"
    prior_path, p11_path, ref_path = P28_EVAL / "summary.json", P11 / "summary.json", REFERENCES / "manifest.json"
    inputs, prior, p11, reference = map(from_json, (input_path, prior_path, p11_path, ref_path))
    protected.update(inputs["source_sha256"])
    protected.update(prior["source_sha256"])
    protected.update(p39["input_sha256"])
    protected.update({
        path.relative_to(OUT).as_posix(): sha(path)
        for path in (p39_path, p38_path, history38, input_path, prior_path, p11_path, ref_path)
    })
    protected[p38_model.relative_to(OUT).as_posix()] = p38["model_sha256"]
    if blend_rule is not None:
        if blend_rule != p39["candidates"][p39["winner"]]:
            raise ValueError("P39 recipe differs from its A/B decision")
        names = [f"sequence_{seed}.pt" for seed in p39["seeds"]]
        if len(set(names)) != 3 or set(names) != set(p39["model_sha256"]) or p39["final_fits"] != len(names):
            raise ValueError("Expected three frozen network members")
        model = tuple(p39_path.with_name(name) for name in names)
        protected.update({path.relative_to(OUT).as_posix(): p39["model_sha256"][path.name] for path in model})
        excluded = p39["excluded_by_variant"][blend_rule["variant"]]
        if p39["included_channels"] != [group for group in GROUPS if group not in excluded]:
            raise ValueError("P39 included channels differ from its A/B decision")
        channels = channel_indices(p39["included_channels"])
        control = SequenceReference(p38_eval, p38_model, p38_channels, p38["selected"])
        candidate, counterpart = "p39", "p38"
    else:
        if p39["winner"] != "p38" or p39["final_fits"] != 0:
            raise ValueError("Expected P38 retention and no new model fit")
        p37_path, history37 = BLEND / "selection.json", BLEND / "evaluation/summary.json"
        p37, p37_eval = map(from_json, (p37_path, history37))
        anchor37 = {"winner": p37["winner"], "rule": p37["selected"]}
        if p37_eval["blend_selection"] != anchor37 or p37["selected"]["branch"] != "p34":
            raise ValueError("Expected the frozen P37 four-hour choice")
        p37_model = CADENCE / "fitted/sequence.pt"
        protected[p37_model.relative_to(OUT).as_posix()] = p37_eval["training"]["model_sha256"]
        protected[history37.relative_to(OUT).as_posix()] = sha(history37)
        control = SequenceReference(p37_eval, p37_model, rule=p37["selected"])
        model, channels, blend_rule = p38_model, p38_channels, p38["selected"]
        candidate, counterpart = "p38", "p37"
    verify_sources(protected)
    columns = pd.read_parquet(AB_FEATURES).columns.tolist()
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="seed-evaluation-") as temp:
        stage = Path(temp)
        kwargs = {"blend_rule": blend_rule, "fixed_control": control, "channels": channels}
        c, c_ci = None, None
        deliverable = stage / "evaluation"
        deliverable.mkdir()
        if candidate == "p39":
            c, masks, c_ci = set_c(
                stage, model, blend_rule["weight"], routed, gcs, rule, prior, reference, p11, True, None, **kwargs,
            )
            masks.to_csv(deliverable / "set_c_paired_replicates.csv", index=False)
        anchor = Ensemble(routed, gcs, rule)
        rows, raw = external(
            stage, model, blend_rule["weight"], anchor, (elastic, reduced), prior, columns, extras, True, None, **kwargs,
        )
        if raw != p38_eval["eicu_raw_sha256"]:
            raise ValueError("The Demo source differs from the frozen P38 evaluation")
        if candidate == "p38":
            previous = {(row["alignment"], row["mode"]): row["metrics"] for row in p38_eval["eicu"]}
            for row in rows:
                before = previous[row["alignment"], row["mode"]]
                np.testing.assert_allclose(
                    [row["metrics"][key] for key in METRICS], [before[key] for key in METRICS], rtol=0, atol=1e-12,
                )
        verify_sources(protected)
        save_json(deliverable / "summary.json", {
            "training": p39, "candidate": candidate, "paired_control": counterpart, "set_c": c, "set_c_clean_ci": c_ci,
            "eicu": rows, "source_sha256": protected, "eicu_raw_sha256": raw,
        })
        deliverable.rename(output)
    print(f"Completed frozen CPU {candidate} versus {counterpart} evaluation at {output}", flush=True)


if __name__ == "__main__":
    evaluate()
