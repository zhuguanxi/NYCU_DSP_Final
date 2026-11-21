"""Run the pinned mortality author schedules serially and resume saved epochs."""

import argparse
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import run_cpu
from author_cpu_patches import install

SUITE = run_cpu.ROOT / "complete_author_schedules"
ORDER = ("ATENet", "Warpformer", "WaveGNN", "ViTST")
TAG = "recoverable_paper_20261121"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    staged = path.with_suffix(".pending")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(staged, path)


def free_gib():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in (
                        "total", "available", "page_total", "page_available",
                        "virtual_total", "virtual_available", "extended")]]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("Cannot inspect free physical memory")
    return memory.available / 1024 ** 3


def prepare(model):
    directory = run_cpu.ROOT / model / TAG
    plan = directory / "plan.json"
    if plan.exists():
        result = json.loads(plan.read_text(encoding="utf-8"))
        if result["source_commit"] != run_cpu.MODELS[model]:
            raise ValueError("Prepared author source revision changed")
        for path, expected in result["code_sha256"].items():
            if sha(directory / "code" / path) != expected:
                raise ValueError(f"Prepared program changed: {path}")
        return directory, result
    recovering_wave_preparation = directory.exists() and model == "WaveGNN"
    if recovering_wave_preparation:
        workspace = directory / "code"
        if "full_epoch.pt" not in (workspace / "utils/trainer.py").read_text(encoding="utf-8"):
            raise ValueError("Incomplete WaveGNN work copy requires inspection")
    else:
        if directory.exists():
            raise FileExistsError(f"Incomplete preparation requires inspection: {directory}")
        directory, workspace = run_cpu.prepare(model, "paper_schedule", TAG)
    if model == "ViTST":
        # Reuse the already verified first-fold images. Remaining folds are generated
        # with their own training-derived scales by the author's image program.
        previous = directory.parent / "cpu_first_fold_01" / "code" / "dataset/P12data/processed_data"
        paths = list(previous.glob("*split0_images"))
        if len(paths) != 1 or len(list(paths[0].glob("*.png"))) != 11988:
            raise ValueError("Expected all 11988 previously generated first-fold images")
        target = workspace / "dataset/P12data/processed_data" / paths[0].name
        shutil.copytree(paths[0], target, copy_function=os.link)
    tasks = []
    original_step = run_cpu.step
    def collect(run_dir, cwd, name, arguments):
        tasks.append({"cwd": str(cwd), "name": name, "arguments": list(map(str, arguments))})
    run_cpu.step = collect
    try:
        {
            "ATENet": run_cpu.run_atenet, "Warpformer": run_cpu.run_warpformer,
            "WaveGNN": run_cpu.run_wavegnn, "ViTST": run_cpu.run_vitst,
        }[model](directory, workspace, "paper_schedule")
    finally:
        run_cpu.step = original_step
    if not recovering_wave_preparation:
        install(model, workspace)
    code_files = [path for path in workspace.rglob("*.py") if "baselines" not in path.parts]
    for path in code_files:
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    result = {
        "model": model, "device": "cpu", "source_commit": run_cpu.MODELS[model],
        "created_utc": now(), "tasks": tasks,
        "code_sha256": {path.relative_to(workspace).as_posix(): sha(path) for path in code_files},
    }
    save(plan, result)
    return directory, result


def verify_output(model, task, log):
    text = log.read_text(encoding="utf-8", errors="replace")
    if model == "ATENet":
        rows = re.findall(r"Testing: AUROC = ([\d.]+) \| AUPRC = ([\d.]+)", text)
        if len(rows) != 5:
            raise ValueError("ATENet did not finish five fold tests")
        return [{"auroc_percent": float(a), "ap_percent": float(b)} for a, b in rows]
    if model == "Warpformer":
        rows = re.findall(r"\-\( Testing \).*?AUROC:\s*([\d.]+), AUPRC:\s*([\d.]+)", text)
        if len(rows) != 1:
            raise ValueError("Warpformer did not finish its held-out test")
        return {"auroc": float(rows[0][0]), "ap": float(rows[0][1])}
    if model == "WaveGNN":
        rows = re.findall(r"Test Metrics: (\{[^\n]+\})", text)
        if len(rows) != 3:
            raise ValueError("WaveGNN did not finish three repetition tests")
        return [ast.literal_eval(row) for row in rows]
    if task["name"] == "images":
        if "Save data in ImageDict_list.npy" not in text:
            raise ValueError("ViTST image preparation did not finish")
        return {"images_complete": True}
    rows = re.findall(r"Testing: AUROC = ([\d.]+) \| AUPRC = ([\d.]+)", text)
    if len(rows) != 5:
        raise ValueError("ViTST did not finish five fold tests")
    return [{"auroc_percent": float(a), "ap_percent": float(b)} for a, b in rows]


def execute(model, directory, task):
    receipt = directory / f"{task['name']}.complete.json"
    if receipt.exists():
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if sha(Path(record["log"])) != record["log_sha256"]:
            raise ValueError("A completed run log changed")
        return record
    attempts = len(list(directory.glob(f"{task['name']}.attempt*.json")))
    command = [sys.executable, "-u", *task["arguments"]]
    env = run_cpu.environment(directory)
    for attempt in range(attempts + 1, attempts + 4):
        # Retries use the latest complete epoch/optimizer/RNG state in the same work copy.
        log = directory / f"{task['name']}.attempt{attempt}.log"
        record = {"command": command, "cwd": task["cwd"], "started_utc": now(), "log": str(log)}
        save(SUITE / "status.json", {"state": "running", "model": model, "task": task["name"], **record})
        print(f"{now()} {model} {task['name']} attempt {attempt}", flush=True)
        with log.open("xb") as output:
            completed = subprocess.run(command, cwd=task["cwd"], env=env, stdout=output, stderr=subprocess.STDOUT)
        record.update({"exit_code": completed.returncode, "ended_utc": now(), "log_sha256": sha(log)})
        try:
            if completed.returncode:
                raise RuntimeError(f"Author process exited {completed.returncode}")
            # After epoch-resume, completed folds can be logged again. The last successful
            # full invocation must still produce all of the model's expected held-out tests.
            record["metrics"] = verify_output(model, task, log)
        except (ValueError, RuntimeError) as error:
            record["failure"] = str(error)
        save(directory / f"{task['name']}.attempt{attempt}.json", record)
        if "failure" not in record:
            save(receipt, record)
            return record
        print(f"{model}: {record['failure']}. See {log}", flush=True)
        if attempt < attempts + 3:
            time.sleep(30)
    raise RuntimeError(f"Three unsuccessful attempts for {model}; inspect its saved logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=ORDER, default=list(ORDER))
    args = parser.parse_args()
    SUITE.mkdir(exist_ok=True)
    if args.prepare_only:
        for model in args.models:
            directory, plan = prepare(model)
            print(model, directory, [job["name"] for job in plan["tasks"]], flush=True)
        return
    import msvcrt
    with (SUITE / "worker.lock").open("a+b") as lock:
        lock.seek(0)
        if lock.read(1) == b"":
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            kedgn = run_cpu.ROOT / "KEDGN/cpu_paper_01/train.log"
            if len(re.findall(r"Testing: AUROC", kedgn.read_text(encoding="utf-8"))) != 5:
                raise ValueError("KEDGN's completed five-seed results are missing")
            for model in args.models:
                minimum = 2
                while free_gib() < minimum:
                    save(SUITE / "status.json", {
                        "state": "waiting_for_memory", "model": model, "checked_utc": now(),
                        "minimum_free_gib": minimum, "available_gib": free_gib(),
                    })
                    time.sleep(30)
                directory, plan = prepare(model)
                for job in plan["tasks"]:
                    execute(model, directory, job)
            save(SUITE / "status.json", {
                "state": "author_schedules_complete", "ended_utc": now(), "models": args.models,
                "next_stage": "Fit the same architectures on P12 A/B before evaluating Set C and eICU Demo",
            })
        except Exception as error:
            save(SUITE / "status.json", {"state": "needs_inspection", "error": str(error), "utc": now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
