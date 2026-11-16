"""Replay the frozen CPU sequence model on a temporary staged cohort."""

import argparse
from pathlib import Path

import numpy as np
import torch

from train_hourly_sequence import THREADS, masked, predict


def evaluate(model_path, input_path, output_path, four_hour=False):
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    model = torch.jit.load(str(model_path), map_location="cpu")
    with np.load(input_path, allow_pickle=False) as packet:
        dynamic, static = packet["cubes"], packet["statics"]
        if four_hour:
            from train_sequence_cadence import cadence_cube

            dynamic = cadence_cube(dynamic) if dynamic.ndim == 3 else np.stack([cadence_cube(x) for x in dynamic])
        if "hidden" in packet.files:
            hidden = packet["hidden"]
            baseline = predict(model, torch.from_numpy(dynamic), torch.from_numpy(static))
            rows = []
            for case in hidden:
                changed = case.any(axis=1)
                scores = baseline.copy()
                if changed.any():
                    altered = masked(dynamic[changed], case[changed])
                    scores[changed] = predict(model, torch.from_numpy(altered), torch.from_numpy(static[changed]))
                rows.append(scores)
        else:
            rows = [predict(model, torch.from_numpy(x), torch.from_numpy(s)) for x, s in zip(dynamic, static, strict=True)]
    np.save(output_path, np.stack(rows), allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--four-hour", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, args.inputs, args.output, args.four_hour)
