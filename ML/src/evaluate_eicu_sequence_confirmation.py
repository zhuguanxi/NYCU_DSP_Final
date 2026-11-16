"""Evaluate P38 as a secondary comparison on the frozen P29 eICU cohort."""

import argparse
from pathlib import Path

import numpy as np

from eicu_demo_features import DEMO, STAY
from evaluate_eicu_confirmation import evaluate, temporal_extension
from evaluate_hourly_sequence import ABLATION, GROUPS, ID, MINUTE, VALUE, hourly_cube, infer, static_values
from prepare import OUT, sha
from sequence_blending import blend
from train_predictive_boosting import from_json, verify_sources

CAP = 6000


class FrozenP38:
    def __init__(self):
        path = ABLATION / "fitted/manifest.json"
        reference_path = ABLATION / "evaluation/summary.json"
        fit = from_json(path)
        reference = from_json(reference_path)
        self.recipe = fit["selected"]
        expected = {"ablation": "without_both", "mode": "log_odds", "weight": 0.25, "bound": 2.0}
        included = [group for group in GROUPS if group not in {"Urine", "RespRate"}]
        if (
            reference["training"] != fit or self.recipe != expected
            or (fit["dataset"], fit["device"], fit["block_hours"]) != ("P12 A/B", "cpu", 4)
            or fit["included_channels"] != included
        ):
            raise ValueError("Expected the evaluated A/B-frozen P38 sequence recipe")
        self.model = path.with_name("sequence.pt")
        self.channels = [GROUPS.index(group) for group in included]
        self.source_sha256 = {
            **fit["input_sha256"],
            path.relative_to(OUT).as_posix(): sha(path),
            reference_path.relative_to(OUT).as_posix(): sha(reference_path),
            self.model.relative_to(OUT).as_posix(): fit["model_sha256"],
        }
        verify_sources(self.source_sha256)
        self.cubes = []

    def features(self, events, index):
        sequence = events.rename(columns={STAY: ID, "minute": MINUTE, "value": VALUE})
        self.cubes.append((index.copy(), hourly_cube(sequence, index)[:, self.channels]))
        return temporal_extension(events, index)

    def predict(self, stage, frames, p28):
        if tuple(frames) != tuple(p28):
            raise ValueError("The frozen P28 scores and sequence frames have different sampling modes")
        cubes, statics = [], []
        for frame, (index, cube) in zip(frames.values(), self.cubes, strict=True):
            if not frame.index.equals(index):
                raise ValueError("P38 stay order differs from the frozen P28 scores")
            cubes.append(cube)
            statics.append(static_values(frame))
        scores = infer(stage, self.model, four_hour=True, cubes=np.stack(cubes), statics=np.stack(statics))
        if scores.shape != (len(frames), len(next(iter(frames.values())))):
            raise ValueError("Unexpected frozen P38 eICU score shape")
        return {mode: blend(p28[mode], values, self.recipe) for mode, values in zip(frames, scores, strict=True)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-demo", action="store_true")
    modes.add_argument("--full-directory", type=Path)
    args = parser.parse_args()
    source = DEMO if args.smoke_demo else args.full_directory.resolve()
    evaluate(source, CAP, args.smoke_demo, ensemble=True, sequence=FrozenP38())
