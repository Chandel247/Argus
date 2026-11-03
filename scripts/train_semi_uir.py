"""Train Semi-UIR under the benchmark protocol, resumable across sessions.

Uses the official per-epoch training/validation code (trainer.Trainer
._train_epoch/_valid_epoch, official losses, AdamP, MultiStepLR, EMA teacher,
reliable bank) and replaces only the outer loop, which upstream cannot resume
and only checkpoints every 20 epochs.

Run directory layout (everything needed to resume or export):
  run/metrics.jsonl     one JSON line per epoch
  run/ckpt/last.pt      full training state (student, teacher, optimizer, ...)
  run/ckpt/best.pt      student weights with the best val PSNR so far
  run/bank/             reliable bank (pseudo-labels for unlabeled images)
  run/DONE              written when all epochs have finished
Checkpoints are written to a temp file and renamed, so readers (the exporter)
never see a partial file.

python scripts/train_semi_uir.py --uieb data/uieb --lsui data/lsui/LSUI --run-dir runs/semi_uir_uieb
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uie.baselines import semi_uir  # noqa: E402

SPLITS = ROOT / "splits"


def read_split(name):
    return [l.strip() for l in (SPLITS / name).read_text().splitlines() if l.strip()]


def atomic_save(obj, path: Path):
    import torch
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def link(src: Path, dst: Path):
    if not dst.exists():
        dst.symlink_to(src.resolve())


def build_data(work: Path, uieb: Path, lsui: Path, bank: Path, smoke: bool):
    """Lay out data the way the official dataset classes expect, using symlinks.
    Every file is named <stem>.png so sorted input/GT/LA/candidate lists align
    (PIL reads by content, not extension). LA maps are computed once and cached."""
    fit, val = read_split("uieb_train_fit.txt"), read_split("uieb_val.txt")
    unl = read_split("unlabeled_lsui800.txt")
    if smoke:
        fit, val, unl = fit[:16], val[:8], unl[:16]

    def la_for(src: Path, dst: Path):
        if not dst.exists():
            Image.fromarray(semi_uir.luminance_estimation(np.asarray(Image.open(src).convert("RGB")))).save(dst)

    groups = [("labeled1", fit, uieb / "raw-890", uieb / "reference-890"),
              ("val", val, uieb / "raw-890", uieb / "reference-890"),
              ("unlabeled", unl, lsui / "input", None)]
    for phase, names, in_dir, gt_dir in groups:
        for sub in ("input", "LA") + (("GT",) if gt_dir else ()):
            (work / phase / sub).mkdir(parents=True, exist_ok=True)
        for n in names:
            stem = Path(n).stem
            link(in_dir / n, work / phase / "input" / f"{stem}.png")
            if gt_dir:
                link(gt_dir / n, work / phase / "GT" / f"{stem}.png")
            la_for(in_dir / n, work / phase / "LA" / f"{stem}.png")
        print(f"  {phase}: {len(names)}")

    # reliable bank, initialised to black images like the official create_candidate.py
    bank.mkdir(parents=True, exist_ok=True)
    blank = Image.new("RGB", (256, 256))
    for n in unl:
        p = bank / f"{Path(n).stem}.png"
        if not p.exists():
            blank.save(p)
    cand = work / "unlabeled" / "candidate"
    if cand.is_symlink() or cand.exists():
        cand.unlink() if cand.is_symlink() else shutil.rmtree(cand)
    cand.symlink_to(bank.resolve(), target_is_directory=True)


class _NullWriter:
    def add_scalar(self, *a, **k): pass
    def add_histogram(self, *a, **k): pass
    def close(self): pass


def cpu_fallback():
    """Upstream hard-codes .cuda(); map it to CPU so the loop can be smoke-tested
    without a GPU."""
    import torch
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self


def rng_state():
    import torch
    s = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    import torch
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--uieb", required=True, type=Path, help="dir with raw-890/ and reference-890/")
    ap.add_argument("--lsui", required=True, type=Path, help="dir with input/ (LSUI)")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--work-dir", type=Path, help="scratch for data layout (default: <run-dir>/../_work)")
    ap.add_argument("--resume-from", type=Path, help="previous run dir to continue from")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop cleanly before this wall time")
    ap.add_argument("--smoke", action="store_true", help="tiny subsets for a pipeline check")
    args = ap.parse_args(argv)
    t_start = time.time()

    import torch
    if not torch.cuda.is_available():
        cpu_fallback()
    semi_uir.activate()
    from torch.utils.data import DataLoader
    from dataset_all import TrainLabeled, TrainUnlabeled, ValLabeled  # official
    from model import AIMnet
    from trainer import Trainer
    from utils import create_emamodel, initialize_weights, setup_seed

    run = args.run_dir
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    if args.resume_from and args.resume_from.resolve() != run.resolve():
        for item in ("ckpt", "bank", "metrics.jsonl"):
            src = args.resume_from / item
            if src.is_dir():
                shutil.copytree(src, run / item, dirs_exist_ok=True)
            elif src.exists():
                shutil.copy2(src, run / item)
    last = run / "ckpt" / "last.pt"
    state = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    if state and state["epoch"] >= args.epochs:
        print(f"already finished ({state['epoch']} epochs)")
        (run / "DONE").touch()
        return

    work = args.work_dir or run.parent / "_work"
    print("preparing data ...")
    build_data(work, args.uieb, args.lsui, run / "bank", args.smoke)

    setup_seed(2022)
    paired = TrainLabeled(dataroot=str(work), phase="labeled1", finesize=args.crop)
    unpaired = TrainUnlabeled(dataroot=str(work), phase="unlabeled", finesize=args.crop)
    val = ValLabeled(dataroot=str(work), phase="val", finesize=args.crop)
    # num_workers/pin_memory are the only additions: they do not change the data
    kw = dict(num_workers=4, pin_memory=torch.cuda.is_available())
    paired_loader = DataLoader(paired, batch_size=args.batch, **kw)
    unpaired_loader = DataLoader(unpaired, batch_size=args.batch, **kw)
    val_loader = DataLoader(val, batch_size=args.val_batch, **kw)

    net, ema = AIMnet(), create_emamodel(AIMnet())
    targs = argparse.Namespace(gpus=args.gpus, crop_size=args.crop, train_batchsize=args.batch,
                               local_rank=-1, num_epochs=args.epochs)
    tr = Trainer(model=net, tmodel=ema, args=targs, supervised_loader=paired_loader,
                 unsupervised_loader=unpaired_loader, val_loader=val_loader,
                 iter_per_epoch=len(unpaired_loader), writer=_NullWriter())
    tr.freeze_teachers_parameters()

    # Upstream creates MUSIQ with as_loss=True, so every reliable-bank scoring
    # call keeps a full autograd graph -- but all three call sites immediately
    # .detach() the scores and only compare them. Scoring under no_grad is
    # numerically identical and avoids OOM on 16 GB GPUs.
    _iqa = tr.iqa_metric

    def iqa_no_grad(x):
        with torch.no_grad():
            return _iqa(x)

    tr.iqa_metric = iqa_no_grad

    best = -1.0
    start = 1
    if state:
        tr.model.module.load_state_dict(state["student"])
        tr.tmodel.load_state_dict(state["teacher"])
        tr.optimizer_s.load_state_dict(state["optimizer"])
        tr.lr_scheduler_s.load_state_dict(state["scheduler"])
        tr.curiter, best, start = state["curiter"], state["best_val_psnr"], state["epoch"] + 1
        set_rng_state(state["rng"])
        print(f"resumed at epoch {start} (best val PSNR {best:.3f})")
    else:
        initialize_weights(tr.model)  # as upstream Trainer.train() does

    epoch_times = []
    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        loss, psnr_train = tr._train_epoch(epoch)
        val_psnr = float(np.mean(tr._valid_epoch(epoch)))
        dt = time.time() - t0
        epoch_times.append(dt)

        rec = {"epoch": epoch, "val_psnr": val_psnr, "train_psnr": float(np.mean(psnr_train)),
               "loss": float(loss), "lr": tr.lr_scheduler_s.get_last_lr()[0], "sec": round(dt, 1),
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if torch.cuda.is_available():
            rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        improved = val_psnr > best
        if improved:
            best = val_psnr
            atomic_save({"student": tr.model.module.state_dict(), "epoch": epoch, "val_psnr": val_psnr,
                         "model": semi_uir.NAME, "upstream_commit": semi_uir.COMMIT},
                        run / "ckpt" / "best.pt")
        rec["best"] = improved
        atomic_save({"student": tr.model.module.state_dict(), "teacher": tr.tmodel.state_dict(),
                     "optimizer": tr.optimizer_s.state_dict(), "scheduler": tr.lr_scheduler_s.state_dict(),
                     "epoch": epoch, "curiter": tr.curiter, "best_val_psnr": best, "rng": rng_state()},
                    last)
        with open(run / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[epoch {epoch}] val PSNR {val_psnr:.3f} (best {best:.3f}) {dt:.0f}s", flush=True)

        elapsed = (time.time() - t_start) / 3600
        if epoch < args.epochs and elapsed + 1.3 * np.mean(epoch_times[-5:]) / 3600 > args.max_hours:
            print(f"time budget reached after epoch {epoch} ({elapsed:.2f}h); resume in a new session")
            return

    (run / "DONE").touch()
    print("training finished")


if __name__ == "__main__":
    main()
