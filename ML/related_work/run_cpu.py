"""Run the pinned related-work code in per-paper CPU work copies."""

import argparse
import csv
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


ML = Path(__file__).resolve().parents[1]
ROOT = ML / "artifacts" / "related_work"
PAPER = ROOT / "paper_data"
DATA = ML.parent / "data" / "predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0"
MODELS = {
    "WaveGNN": "2bd2c3c899db650c28658764a86224318cd7ec1c",
    "ATENet": "dc3149ca60f02dafd8e073b336479f280efa51e7",
    "Warpformer": "377d64db2b146c7acf4197c970a236dca3ba0663",
    "ViTST": "7774e593754e8ace77e7f1e14097f3d2705c6303",
    "KEDGN": "c82bc004cd1d1305fc58ca7150369d78682fccad",
}


def replace(path, old, new):
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError(f"Patch anchor absent in {path.name}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def ensure_parent(path):
    path.parent.mkdir(exist_ok=True, parents=True)


def link(source, target):
    ensure_parent(target)
    if target.exists():
        if not source.samefile(target):
            if source.read_bytes() != target.read_bytes():
                raise FileExistsError(target)
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    os.link(source, target)


def environment(run_dir):
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true", "PYTHONPATH": os.pathsep.join(map(str, (ROOT / "cpu_packages", ML / "related_work"))),
        "HF_HOME": str(ROOT / "huggingface"), "MPLCONFIGDIR": str(run_dir / "mplconfig"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "PYTHONUTF8": "1", "MPLBACKEND": "Agg",
    })
    return env


def smoke_indices(labels, size):
    import numpy as np

    y = np.asarray(labels).reshape(-1)
    if set(y) != {0, 1}:
        raise ValueError("Expected binary mortality labels")
    rng = np.random.default_rng(42)
    selected = np.concatenate([rng.choice(np.flatnonzero(y == k), size=size // 2, replace=False) for k in (0, 1)])
    return rng.permutation(selected)


def smoke_arrays(train, val, test, ytrain, yval, ytest):
    import numpy as np

    positions = [smoke_indices(y, size) for y, size in zip((ytrain, yval, ytest), (16, 8, 8), strict=True)]
    subset = [np.asarray(rows)[idx] for rows, idx in zip((train, val, test, ytrain, yval, ytest), positions * 2, strict=True)]
    print("Smoke subsets: train 16, validation 8, test 8", flush=True)
    return tuple(subset)


def step(run_dir, cwd, name, arguments):
    command = [sys.executable, "-u", *map(str, arguments)]
    started = time.time()
    print(f"{name} started: {run_dir.name}", flush=True)
    log = run_dir / f"{name}.log"
    with log.open("xb") as output:
        result = subprocess.run(command, cwd=cwd, env=environment(run_dir), stdout=output, stderr=subprocess.STDOUT)
    print(f"{name}: exit {result.returncode} in {time.time()-started:.1f}s; {log}", flush=True)
    if result.returncode:
        raise RuntimeError(f"{name} failed. Read {log}")


def version(upstream):
    with (upstream / ".git" / "HEAD").open(encoding="ascii") as f:
        head = f.read().strip()
    if not head.startswith("ref: "):
        return head
    ref = upstream / ".git" / head.removeprefix("ref: ")
    return ref.read_text(encoding="ascii").strip()


def prepare(model, profile, run_name):
    upstream = ROOT / model / "upstream"
    if version(upstream) != MODELS[model]:
        raise ValueError("The author checkout differs from the tested version")
    run_dir = ROOT / model / run_name
    workspace = run_dir / "code"
    shutil.copytree(upstream, workspace, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.log"))
    (run_dir / "run.json").write_text(json.dumps({
        "author_commit": MODELS[model], "device": "cpu", "profile": profile, "model": model,
    }, indent=2), encoding="utf-8")
    return run_dir, workspace


def aten_inputs(workspace):
    target = workspace / "dataset" / "P12data"
    source = PAPER / "P12data"
    names = ["processed_data/PTdict_list.npy", "processed_data/arr_outcomes.npy"]
    names.extend(f"splits/phy12_split{i}.npy" for i in range(1, 6))
    for name in names:
        link(source / name, target / name)


def run_atenet(run_dir, workspace, profile):
    aten_inputs(workspace)
    for file in ("main.py", "models.py", "subutils.py"):
        replace(workspace / file, ".cuda()", ".cpu()")
    if profile == "smoke":
        replace(workspace / "main.py", "num_epochs = 20", "num_epochs = 1")
        replace(workspace / "main.py", "n_splits = 5", "n_splits = 1")
        replace(workspace / "main.py", "predictive_label='mortality')",
                "predictive_label='mortality')\n        from run_cpu import smoke_arrays\n"
                "        Ptrain, Pval, Ptest, ytrain, yval, ytest = smoke_arrays(Ptrain, Pval, Ptest, ytrain, yval, ytest)")
        replace(workspace / "main.py", "            if strategy == 1:\n                n_batches",
                "            batch_size = 8\n            if strategy == 1:\n                n_batches")
    (workspace / "models").mkdir(exist_ok=True)
    (workspace / "results" / "log").mkdir(exist_ok=True, parents=True)
    (workspace / "results" / "fig").mkdir(exist_ok=True, parents=True)
    step(run_dir, workspace, "train", [
        "main.py", "--dataset", "P12", "--predictive_label", "mortality", "--learn-emb", "--hidden", "32",
        "--embed-time", "128", "--num-heads", "2", "--lr", "0.0001", "--alpha", "0.01", "--beta", "0.1",
    ])


def verify_kedgn_arrays(author, processed):
    import numpy as np

    expected = np.load(author / "arr_outcomes.npy", allow_pickle=True)
    outcomes = np.load(processed / "arr_outcomes.npy", allow_pickle=True)
    samples = np.load(processed / "PTdict_list.npy", allow_pickle=True)
    lookup = {int(row["id"]): (row, y) for row, y in zip(samples, outcomes, strict=True)}
    records, labels = zip(*(lookup[int(row[0])] for row in expected), strict=True)
    np.testing.assert_array_equal(np.stack(labels), expected)
    np.testing.assert_array_equal([int(row["id"]) for row in records], expected[:, 0])
    np.save(processed / "PTdict_list.npy", np.asarray(records, dtype=object))
    np.save(processed / "arr_outcomes.npy", np.asarray(labels))
    print(f"KEDGN: aligned {len(labels)} Set A records and outcomes to the author split order", flush=True)


def run_kedgn(run_dir, workspace, profile):
    base = workspace / "data" / "physionet"
    scripts = base / "process_scripts"
    raw = base / "raw"
    files = sorted((DATA / "set-a").glob("*.txt"))
    if len(files) != 4000:
        raise ValueError("Expected all 4000 official Set A stays")
    for source in files:
        link(source, raw / "set-a" / source.name)
    link(DATA / "Outcomes-a.txt", raw / "Outcomes-a.txt")
    replace(scripts / "IrregularSampling.py", "exit(-1)", "exit(0)")
    for script in ("ParseData.py", "IrregularSampling.py", "remove_outliers.py"):
        step(run_dir, scripts, Path(script).stem, [script])
    verify_kedgn_arrays(ROOT / "KEDGN/upstream/data/physionet/processed_data", base / "processed_data")
    for file in ("train.py", "utils.py"):
        replace(workspace / file, ".cuda()", ".cpu()")
    replace(workspace / "train.py", "args.plm + suffix).to(device)", "args.plm + suffix, map_location='cpu').to(device)")
    if profile != "paper_schedule":
        replace(workspace / "train.py", "for k in range(5):", "for k in range(1):")
    if profile == "smoke":
        replace(workspace / "train.py", "get_data_split(base_path, split_path, dataset=dataset)",
                "get_data_split(base_path, split_path, dataset=dataset)\n    from run_cpu import smoke_arrays\n"
                "    Ptrain, Pval, Ptest, ytrain, yval, ytest = smoke_arrays(Ptrain, Pval, Ptest, ytrain, yval, ytest)")
    step(run_dir, workspace, "train", [
        "train.py", "--dataset", "physionet", "--cuda", "-1", "--epochs", "10" if profile == "paper_schedule" else "1",
        "--batch_size", "8" if profile == "smoke" else "256", "--lr", "0.001", "--plm", "bert", "--plm_rep_dim", "768", "--query_vector_dim",
        "5", "--node_emb_dim", "9", "--rarity_alpha", "1", "--hidden_dim", "8", "--source", "gpt",
    ])


def run_warpformer(run_dir, workspace, profile):
    raw = workspace / "local_data" / "physionet" / "PhysioNet" / "raw"
    raw.mkdir(parents=True)
    link(DATA / "Outcomes-a.txt", raw / "Outcomes-a.txt")
    source = DATA / "set-a"
    if len(list(source.glob("*.txt"))) != 4000:
        raise ValueError("Expected 4000 Set A records")
    selected = warp_smoke_ids(workspace) if profile == "smoke" else None
    with tarfile.open(raw / "set-a.tar.gz", "w:gz") as archive:
        if selected is None:
            archive.add(source, arcname="set-a")
        else:
            for key in selected:
                filename = f"{key}.txt"
                archive.add(source / filename, arcname=f"set-a/{filename}")
    replace(workspace / "data_utils" / "physionet.py", "set-a.tar.gz?download", "set-a.tar.gz")
    replace(workspace / "Main_warp.py", "opt.device = torch.device('cuda')", "opt.device = torch.device('cpu')")
    if profile == "smoke":
        path = workspace / "Dataset_MM.py"
        replace(path, "model_selection.train_test_split(train_data, train_size=0.8,",
                "model_selection.train_test_split(train_data, train_size=16,")
    out = run_dir / "models"
    out.mkdir()
    for seed in range(1 if profile == "smoke" else 5):
        step(run_dir, workspace, f"train_seed_{seed}", [
            "Main_warp.py", "--data_path", "local_data/", "--batch", "32", "--lr", "1e-3",
            "--epoch", "1" if profile == "smoke" else "50", "--patience", "5", "--log", "../models/",
            "--save_path", "../models/", "--task", "physio", "--seed", seed, "--warp_num", "0_0.2_1",
            "--batch_size", "4" if profile == "smoke" else "32", "--d_inner_hid", "64", "--d_k", "8", "--d_model", "64", "--d_v", "8",
            "--dropout", "0.0", "--n_head", "1", "--n_layers", "3",
        ])


def warp_smoke_ids(workspace):
    with (DATA / "Outcomes-a.txt").open(newline="", encoding="utf-8") as source:
        labels = {int(row["RecordID"]): int(row["In-hospital_death"]) for row in csv.DictReader(source)}
    paths = [workspace / "data_utils" / f"{name}_record_id.pkl" for name in ("train", "test")]
    train, test = [pickle.loads(path.read_bytes()) for path in paths]
    boundary = int(0.8 * len(train))
    subsets = []
    for rows, size in zip((train[:boundary], train[boundary:], test), (16, 8, 8), strict=True):
        indices = smoke_indices([labels[int(key)] for key in rows], size)
        subsets.extend(int(rows[i]) for i in indices)
    if len(set(subsets)) != 32:
        raise ValueError("Warpformer author split selected overlapping records")
    print("Warpformer smoke archive: 16 author-train, 8 author-validation, 8 author-test", flush=True)
    return subsets


def run_wavegnn(run_dir, workspace, profile):
    overrides = ["--dropout", "0.1"]
    if profile == "smoke":
        main = workspace / "run.py"
        replace(main, "            stats = train_dataset.get_dims()\n\n", "")
        replace(main, "            # create dataloaders", """            stats = train_dataset.get_dims()
            from run_cpu import smoke_indices
            from torch.utils.data import Subset
            train_dataset, val_dataset, test_dataset = [
                Subset(ds, smoke_indices(ds.y, size))
                for ds, size in zip((train_dataset, val_dataset, test_dataset), (8, 4, 4), strict=True)
            ]
            print("Smoke subsets: train 8, validation 4, test 4", flush=True)
            # create dataloaders""")
        overrides = ["--gradient_accumulation_step", "1"]
    step(run_dir, workspace, "train", [
        "run.py", "--dataset", "P12", "--device", "cpu", "--main_path", PAPER.as_posix() + "/",
        "--save_dir", (run_dir / "models").as_posix(), "--epochs", "1" if profile == "smoke" else "20",
        "--n_runs", "1" if profile == "smoke" else "3", *overrides,
    ])


def reuse_vitst_grid(path):
    replace(path, "def draw_image(pid,", """from functools import lru_cache

@lru_cache(maxsize=1)
def reusable_grid(grid_height, grid_width, img_height, img_width, dpi):
    fig, axes = plt.subplots(grid_height, grid_width, squeeze=False,
                             figsize=(img_width / dpi, img_height / dpi), dpi=dpi, frameon=False)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    return fig, axes.ravel()

def draw_image(pid,""")
    replace(path, "    drawed_params = []", """    fig, axes = reusable_grid(grid_height, grid_width, img_height, img_width, dpi)
    lines = []
    drawed_params = []""")
    replace(path, "        plt.subplot(grid_height, grid_width, idx+1) # 6*6", "        ax = axes[idx]")
    replace(path, "            plt.plot(", "            line, = ax.plot(")
    replace(path, "        else:\n            line, = ax.plot(", "        else:\n            ax.set_prop_cycle(None)\n            line, = ax.plot(")
    for name in ("xlim", "ylim", "xticks", "yticks"):
        replace(path, f"        plt.{name}(", f"        ax.set_{name}(")
    replace(path, "        drawed_params.append(param)", "        lines.append(line)\n        drawed_params.append(param)")
    replace(path, """    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(img_path, pad_inches=0)
    plt.clf()""", """    axes[len(lines) - 1].margins(0, 0)
    fig.savefig(img_path, pad_inches=0)
    for line in lines:
        line.remove()""")


def run_vitst(run_dir, workspace, profile):
    import numpy as np

    aten_inputs(workspace)
    dataset = workspace / "dataset" / "P12data"
    source = PAPER / "P12data" / "processed_data"
    link(source / "ts_params.npy", dataset / "processed_data" / "ts_params.npy")
    images = dataset / "process_scripts" / "ConstructImage.py"
    replace(images, 'base_path = "../processed_data/" + base_path',
            'base_path = "../processed_data/" + base_path.replace("*", "x")')
    if profile != "smoke":
        reuse_vitst_grid(images)
    if profile != "paper_schedule":
        replace(images, "for split_idx in range(5):", "for split_idx in range(1):")
    if profile == "smoke":
        replace(images, """        # second round, draw the image for each patient
        for idx, p in tqdm(enumerate(Pdict_list)):""", """        # second round, draw a bounded subset from the original split
        from run_cpu import smoke_indices
        labels = arr_outcomes[:, -1]
        samples = [part[smoke_indices(labels[part], size)]
                   for part, size in zip((idx_train, idx_val, idx_test), (8, 4, 4), strict=True)]
        selected = np.concatenate(samples)
        for idx in tqdm(selected):
            p = Pdict_list[idx]""")
    step(run_dir, images.parent, "images", [images.name])
    if profile == "smoke":
        # Replace hardlinks atomically after generating images so shared source arrays remain unchanged.
        labels_path = dataset / "processed_data" / "arr_outcomes.npy"
        split_path = dataset / "splits" / "phy12_split1.npy"
        labels = np.load(labels_path, allow_pickle=True)
        split = np.load(split_path, allow_pickle=True)
        selected = np.concatenate([part[smoke_indices(labels[part, -1], size)]
                                   for part, size in zip(split, (8, 4, 4), strict=True)])
        samples = np.load(dataset / "processed_data" / "ImageDict_list.npy", allow_pickle=True)
        np.testing.assert_array_equal([int(row["id"]) for row in samples], labels[selected, 0])
        mini_split = np.asarray([np.arange(0, 8), np.arange(8, 12), np.arange(12, 16)], dtype=object)
        for path, data in ((labels_path, labels[selected]), (split_path, mini_split)):
            temp = path.with_suffix(".new.npy")
            np.save(temp, data)
            temp.replace(path)

    paths = list((dataset / "processed_data").glob("*split0_images"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one P12 image setting")
    prefix = paths[0].name.removesuffix("split0_images")
    train = workspace / "code" / "Vision-Text" / "run_VisionTextCLS.py"
    replace(train, "sys.path.insert(0, '/mnt/raid0/zekun/ViTST/code')",
            "from pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))")
    replace(train, "from transformers import *",
            "from transformers import AutoFeatureExtractor, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments")
    replace(train, "    max_length = 64", "    image_model_path = args.image_model_path or image_model_path\n"
            "    text_model_path = args.text_model_path or text_model_path\n    max_length = 64")
    replace(train, "    # eval_dataset=val_dataset,\n    eval_dataset=test_dataset,",
            "    eval_dataset=val_dataset,")
    # Scikit-learn matches the binary scores and avoids dynamic Hub metric code.
    code = train.read_text(encoding="utf-8")
    start = code.index('        metric = load_metric("accuracy")')
    end = code.index("        denoms = ", start)
    code = code[:start] + """        predicted = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predicted)
        precision = precision_score(labels, predicted, zero_division=0)
        recall = recall_score(labels, predicted, zero_division=0)
        f1 = f1_score(labels, predicted, zero_division=0)

""" + code[end:]
    train.write_text(code, encoding="utf-8")
    if profile == "smoke":
        replace(train, "gradient_accumulation_steps=4,", "gradient_accumulation_steps=1,")
    settings = run_dir.parent / "pretrained"
    if any(not (settings / name / "model.safetensors").is_file() for name in ("swin", "roberta")):
        raise FileNotFoundError("Download and verify the pinned public Swin and RoBERTa weights first")
    step(run_dir, train.parent, "train", [
        train.name, "--dataset", "P12", "--dataset_prefix", prefix, "--image_model", "swin",
        "--image_model_path", settings / "swin", "--text_model", "roberta", "--text_model_path", settings / "roberta",
        "--output_dir", "local_cpu", "--do_train", "--epochs", "4",
        "--n_splits", "5" if profile == "paper_schedule" else "1",
        "--train_batch_size", "2" if profile == "smoke" else "48",
        "--eval_batch_size", "2" if profile == "smoke" else "128",
        "--logging_steps", "1" if profile == "smoke" else "20",
        "--save_steps", "1" if profile == "smoke" else "100",
        "--freeze_vision_model", "True" if profile == "smoke" else "False",
        "--freeze_text_model", "True" if profile == "smoke" else "False",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runners = {"ATENet": run_atenet, "KEDGN": run_kedgn, "Warpformer": run_warpformer, "WaveGNN": run_wavegnn,
               "ViTST": run_vitst}
    parser.add_argument("model", choices=tuple(runners))
    parser.add_argument("--profile", choices=("smoke", "first_epoch", "first_fold", "paper_schedule"), default="smoke")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag.replace("_", "").isalnum():
        raise ValueError("Run tag must contain only letters, numbers and underscores")
    if args.profile == "first_epoch" and args.model != "KEDGN":
        parser.error("first_epoch is available for KEDGN")
    if args.profile == "first_fold" and args.model != "ViTST":
        parser.error("first_fold is available for ViTST")
    run_dir, workspace = prepare(args.model, args.profile, args.tag)
    runners[args.model](run_dir, workspace, args.profile)


if __name__ == "__main__":
    main()
