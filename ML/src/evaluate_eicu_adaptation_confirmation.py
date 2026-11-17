"""Add frozen P45 to the patient-disjoint P29 and P40 eICU comparisons."""

import argparse
from pathlib import Path

import joblib
import numpy as np

from eicu_demo_features import DEMO
from evaluate_eicu_confirmation import evaluate
from evaluate_eicu_sequence_confirmation import CAP, FrozenP38
from evaluate_gcs_routed_temporal import RoutedPredictor
from evaluate_real_eicu_augmentation import OUTPUT as P44
from evaluate_robust_ensemble import Ensemble
from prepare import OUT, sha
from train_predictive_boosting import from_json, verify_sources

P45 = OUT / "p45_augmentation_branch_ablation"


class FrozenP45:
    def __init__(self):
        recipe_path, p44_path = P45 / "summary.json", P44 / "summary.json"
        fixed, p44 = from_json(recipe_path), from_json(p44_path)
        self.p26_rule, self.p28_rule = p44["p26_rule"], p44["p28_rule"]
        self.demo_patient_sha256 = p44["raw_eicu_sha256"]["patient.csv.gz"]
        if (
            (p44["device"], p44["ab_stays"], p44["demo_stays"], p44["extra_weight_per_stay"]) != ("cpu", 8000, 782, 1.0)
            or fixed["branches_from_p44"]["primary_only"] != ["temporal"] or fixed["final_fits"] != 0
            or self.p26_rule != {"primary": "temporal", "fallback": "blind_lgb", "weight": 1.0}
            or fixed["raw_eicu_sha256"] != p44["raw_eicu_sha256"]
        ):
            raise ValueError("Expected the evaluated P45 primary-only adaptation")
        path = P44 / "fitted" / "temporal.joblib"
        self.source_sha256 = {
            **fixed["source_sha256"],
            recipe_path.relative_to(OUT).as_posix(): sha(recipe_path),
            p44_path.relative_to(OUT).as_posix(): sha(p44_path),
            path.relative_to(OUT).as_posix(): p44["model_sha256"]["temporal"],
        }
        verify_sources(self.source_sha256)
        self.model = joblib.load(path)
        self.model.set_params(device="cpu", n_jobs=2)
        if self.model.classes_.tolist() != [0, 1] or self.model.get_xgb_params()["device"] != "cpu":
            raise ValueError("Unexpected P45 class order or inference device")
        self.recipe = {
            "adapted_branch": "temporal", "training_stays": {"p12_ab": 8000, "eicu_demo": 782},
            "original_fallback": "blind_lgb", "model_sha256": p44["model_sha256"]["temporal"],
        }

    def predict(self, frames, routed, gcs, rule, columns, p28_scores):
        if routed.rule != self.p26_rule or rule != self.p28_rule or set(frames) != set(p28_scores):
            raise ValueError("P45 requires the same routes, mixture and sampling modes as P28")
        adapted = RoutedPredictor(routed.rule, {**routed.models, "temporal": self.model})
        p45 = Ensemble(adapted, gcs, rule)
        scores, checks = {}, {}
        for mode, frame in frames.items():
            values = p45.predict_proba(frame.loc[:, columns])[:, 1]
            absent = frame["GCS__count"].eq(0).to_numpy()
            base = p28_scores[mode]
            if len(values) != len(base) or not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
                raise ValueError("Unexpected P45 probability values or shape")
            np.testing.assert_array_equal(values[absent], base[absent])
            scores[mode] = values
            checks[mode] = {
                "stays_scored": len(values), "gcs_absent_identical_to_p28": int(absent.sum()),
                "gcs_observed_scores_changed": int(np.count_nonzero(values[~absent] != base[~absent])),
            }
        return scores, checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-demo", action="store_true")
    modes.add_argument("--full-directory", type=Path)
    args = parser.parse_args()
    source = DEMO if args.smoke_demo else args.full_directory.resolve()
    evaluate(source, CAP, args.smoke_demo, ensemble=True, sequence=FrozenP38(), adaptation=FrozenP45())
