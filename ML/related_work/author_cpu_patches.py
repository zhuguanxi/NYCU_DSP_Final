"""Patch verified CPU operations into isolated copies of author programs."""

from pathlib import Path


def change(path, old, new):
    value = path.read_text(encoding="utf-8")
    if value.count(old) != 1:
        raise ValueError(f"Expected one patch anchor in {path}: {old[:90]}")
    path.write_text(value.replace(old, new), encoding="utf-8")


def patch_atenet(workspace):
    path = workspace / "main.py"
    change(path, "import models", """import models
from pathlib import Path
from author_cpu_runtime import bounded_attention, save_epoch, restore_epoch
models.multiTimeAttention.attention = bounded_attention""")
    change(path, "            for epoch in range(num_epochs):", """            resume_path = Path(model_path) / f"resume_fold{split_idx}_run{m}.pt"
            first_epoch = 0
            if resume_path.exists():
                saved = restore_epoch(resume_path, models={"model": model}, optimizer=optimizer, scheduler=scheduler)
                first_epoch = saved["epoch"] + 1
                best_auc_val = saved["metadata"]["best_auc_val"]
                times = saved["metadata"]["times"]
                idx_0 = saved["metadata"]["idx_0"]
                expanded_idx_1 = saved["metadata"]["expanded_idx_1"]
                print(f"Resuming ATENet fold {split_idx} at epoch {first_epoch + 1}", flush=True)
            for epoch in range(first_epoch, num_epochs):""")
    change(path, "            logger.info('Total Time elapsed: %.3f' % (sum(times) / num_epochs))", """                save_epoch(resume_path, epoch=epoch, models={"model": model}, optimizer=optimizer,
                           scheduler=scheduler, metadata={"best_auc_val": best_auc_val, "times": times,
                           "idx_0": idx_0, "expanded_idx_1": expanded_idx_1})
                print(f"ATENet fold {split_idx}, epoch {epoch + 1}/{num_epochs}, "
                      f"validation AP={aupr_val:.6f}, AUROC={auc_val:.6f}", flush=True)
            logger.info('Total Time elapsed: %.3f' % (sum(times) / num_epochs))""")


def patch_warpformer(workspace):
    path = workspace / "Main_warp.py"
    change(path, "eps=1e-7", """from pathlib import Path
from author_cpu_runtime import save_epoch, restore_epoch

eps=1e-7""")
    change(path, "    epoch = 0\n    if not opt.test_only:", """    epoch = 0
    first_epoch = 0
    resume_path = Path(save_path + ".resume.pt")
    if resume_path.exists() and not opt.test_only:
        saved = restore_epoch(resume_path, models={"encoder": model, "classifier": classifier},
                              optimizer=optimizer, scheduler=scheduler)
        first_epoch = saved["epoch"] + 1
        for key, value in saved["metadata"].items():
            setattr(early_stopping, key, value)
        print(f"Resuming Warpformer at epoch {first_epoch + 1}", flush=True)
    if not opt.test_only:""")
    change(path, "        for epoch_i in range(opt.epoch):",
           "        for epoch_i in range(first_epoch, opt.epoch):")
    change(path, "            scheduler.step()", """            scheduler.step()
            save_epoch(resume_path, epoch=epoch_i, models={"encoder": model, "classifier": classifier},
                       optimizer=optimizer, scheduler=scheduler, metadata={
                       key: getattr(early_stopping, key) for key in
                       ("counter", "best_score", "early_stop", "val_loss_min", "best_epoch")})""")


def patch_wavegnn(workspace):
    run = workspace / "run.py"
    change(run, "torch.autograd.set_detect_anomaly(True)", "torch.autograd.set_detect_anomaly(False)")
    trainer = workspace / "utils" / "trainer.py"
    change(trainer, "class Trainer:", """from pathlib import Path
from author_cpu_runtime import save_epoch, restore_epoch


class Trainer:""")
    change(trainer, "        while True:", """        resume_path = Path(self.args.save_dir) / "full_epoch.pt"
        if resume_path.exists():
            saved = restore_epoch(resume_path, models={"model": self.model},
                                  optimizer=self.optimizer, scheduler=self.scheduler)
            e = saved["epoch"] + 1
            meta = saved["metadata"]
            train_loss, eval_loss = meta["train_loss"], meta["eval_loss"]
            patience_count, prev_val_loss = meta["patience_count"], meta["prev_val_loss"]
            self.checkpoint_saver.best_val = meta["best_val"]
            self.best_threshold = meta["best_threshold"]
        while e < epochs and patience_count < patience:""")
    change(trainer, "            if e >= epochs or patience_count >= patience:\n                break",
           """            finished = e >= epochs or patience_count >= patience
            if finished:
                self.save_full_epoch(resume_path, e, train_loss, eval_loss, patience_count, prev_val_loss)
                break""")
    change(trainer, "        return (train_loss, eval_loss)", """            self.save_full_epoch(resume_path, e, train_loss, eval_loss, patience_count, prev_val_loss)
        return (train_loss, eval_loss)

    def save_full_epoch(self, path, e, train_loss, eval_loss, patience_count, prev_val_loss):
        save_epoch(path, epoch=e - 1, models={"model": self.model}, optimizer=self.optimizer,
                   scheduler=self.scheduler, metadata={"train_loss": train_loss, "eval_loss": eval_loss,
                   "patience_count": patience_count, "prev_val_loss": prev_val_loss,
                   "best_val": self.checkpoint_saver.best_val, "best_threshold": self.best_threshold})
        print(f"WaveGNN saved full epoch {e}", flush=True)""")
    # Stable paths are required to find the same run checkpoint after restarting.
    change(run, "args.save_dir = get_save_dir(f\"{args.save_dir}\", training=True)",
           """args.save_dir = f"{args.save_dir}/recoverable"
    os.makedirs(args.save_dir, exist_ok=True)""")
    change(run, "    for run_id in range(args.n_runs):", """    checkpoint_base = args.save_dir
    for run_id in range(args.n_runs):""")
    change(run, "        for split_id in range(args.n_splits):", """        for split_id in range(args.n_splits):
            args.save_dir = os.path.join(checkpoint_base, f"run_{run_id}_split_{split_id}")
            os.makedirs(args.save_dir, exist_ok=True)""")


def patch_vitst(workspace):
    path = workspace / "code" / "Vision-Text" / "run_VisionTextCLS.py"
    change(path, "        train_results = trainer.train()", """        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        train_results = trainer.train(resume_from_checkpoint=last_checkpoint)""")


def install(model, workspace):
    {"ATENet": patch_atenet, "Warpformer": patch_warpformer, "WaveGNN": patch_wavegnn,
     "ViTST": patch_vitst}[model](Path(workspace))
