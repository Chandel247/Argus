"""Train Semi-UIR under the benchmark protocol, resumable across sessions.

Uses the official per-epoch training/validation code (trainer.Trainer
._train_epoch/_valid_epoch, official losses, AdamP, MultiStepLR, EMA teacher,
reliable bank) and replaces only the outer loop, which upstream cannot resume
and only checkpoints every 20 epochs.

Run directory layout (everything needed to resume or export):
  run/metrics.jsonl     one JSON line per epoch
  run/ckpt/last.pt      full training state (student, teacher, optimizer, ...)
  run/ckpt/best.pt      student weights with the best val PSNR so far
  run/bank.pack         reliable bank (pseudo-labels for unlabeled images),
                        one tar, rewritten after every epoch
  run/la.pack           cached illumination maps (computed once, ~1h on Kaggle)
  run/mem.log           host memory telemetry, one JSON line per minute
  run/DONE              written when all epochs have finished
Checkpoints and packs are written to a temp file and renamed, so readers (the
exporter, a fetch) never see a partial file. The bank itself lives in the work
dir while training; older runs that kept it as run/bank/ are still resumable.

python scripts/train_semi_uir.py --uieb data/uieb --lsui data/lsui/LSUI --run-dir runs/semi_uir_uieb
"""
from __future__ import annotations

import argparse
import gc
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

from uie import runstate  # noqa: E402
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


def _la_init():
    import cv2
    cv2.setNumThreads(1)  # parallelism comes from the process pool


def _la_job(paths):
    src, dst = paths
    tmp = Path(str(dst) + ".tmp.png")
    Image.fromarray(semi_uir.luminance_estimation(np.asarray(Image.open(src).convert("RGB")))).save(tmp)
    os.replace(tmp, dst)


def build_data(work: Path, uieb: Path, lsui: Path, bank: Path, smoke: bool):
    """Lay out data the way the official dataset classes expect, using symlinks.
    Every file is named <stem>.png so sorted input/GT/LA/candidate lists align
    (PIL reads by content, not extension). Missing LA maps are computed in
    parallel; existing ones (e.g. restored from la.pack) are kept."""
    fit, val = read_split("uieb_train_fit.txt"), read_split("uieb_val.txt")
    unl = read_split("unlabeled_lsui800.txt")
    if smoke:
        fit, val, unl = fit[:16], val[:8], unl[:16]

    groups = [("labeled1", fit, uieb / "raw-890", uieb / "reference-890"),
              ("val", val, uieb / "raw-890", uieb / "reference-890"),
              ("unlabeled", unl, lsui / "input", None)]
    jobs = []
    for phase, names, in_dir, gt_dir in groups:
        for sub in ("input", "LA") + (("GT",) if gt_dir else ()):
            (work / phase / sub).mkdir(parents=True, exist_ok=True)
        for n in names:
            stem = Path(n).stem
            link(in_dir / n, work / phase / "input" / f"{stem}.png")
            if gt_dir:
                link(gt_dir / n, work / phase / "GT" / f"{stem}.png")
            dst = work / phase / "LA" / f"{stem}.png"
            if not dst.exists():
                jobs.append((in_dir / n, dst))
        print(f"  {phase}: {len(names)}")
    if jobs:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        t0 = time.time()
        # spawn, not fork: the parent may already have initialised CUDA
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 2, mp_context=mp.get_context("spawn"),
                                 initializer=_la_init) as ex:
            list(ex.map(_la_job, jobs, chunksize=4))
        print(f"  computed {len(jobs)} illumination maps in {time.time() - t0:.0f}s")

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
    return [phase for phase, *_ in groups]


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
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers per loader")
    ap.add_argument("--mem-interval", type=float, default=60, help="seconds between memory log lines")
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
    import trainer as upstream_trainer

    def compute_psnr_ssim(recoverd, clean):
        """Upstream utils.compute_psnr_ssim with skimage's removed
        multichannel=True replaced by channel_axis=-1 (same computation)."""
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        r = np.clip(recoverd.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
        c = np.clip(clean.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
        psnr = sum(peak_signal_noise_ratio(c[i], r[i], data_range=1) for i in range(len(r)))
        ssim = sum(structural_similarity(c[i], r[i], data_range=1, channel_axis=-1) for i in range(len(r)))
        return psnr / len(r), ssim / len(r), len(r)

    upstream_trainer.compute_psnr_ssim = compute_psnr_ssim  # trainer did `from utils import *`

    run = args.run_dir
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    resumed_elsewhere = args.resume_from and args.resume_from.resolve() != run.resolve()
    if resumed_elsewhere:
        for item in ("ckpt", "metrics.jsonl", "mem.log", "bank.pack", "la.pack"):
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
    bank = work / "bank"
    memlog = runstate.MemLogger(run / "mem.log", args.mem_interval)
    memlog.log(event="start", resume_epoch=state["epoch"] if state else 0)
    memlog.start()

    print("preparing data ...")
    t_prep = time.time()
    bank.mkdir(parents=True, exist_ok=True)
    if (run / "bank.pack").exists():
        print(f"  bank: {runstate.read_pack(run / 'bank.pack', work)} images from bank.pack")
    else:  # runs from before bank.pack kept the bank as a directory
        for legacy in filter(None, [args.resume_from and args.resume_from / "bank", run / "bank"]):
            if legacy.is_dir():
                n = sum(shutil.copy2(p, bank / p.name) is not None for p in legacy.glob("*.png"))
                print(f"  bank: {n} images from {legacy}")
                break
    if (run / "la.pack").exists():
        print(f"  LA: {runstate.read_pack(run / 'la.pack', work, overwrite=False)} maps from la.pack")
    phases = build_data(work, args.uieb, args.lsui, bank, args.smoke)
    if not (run / "la.pack").exists():
        n = runstate.write_pack({f"{p}/LA": work / p / "LA" for p in phases}, run / "la.pack")
        print(f"  saved {n} LA maps to la.pack")
    if (run / "bank").is_dir():  # migrate: the per-file bank made output fetches unreliable
        shutil.rmtree(run / "bank")
    runstate.write_pack({"bank": bank}, run / "bank.pack")
    print(f"data ready in {time.time() - t_prep:.0f}s")
    memlog.log(event="data_ready")

    setup_seed(2022)
    paired = TrainLabeled(dataroot=str(work), phase="labeled1", finesize=args.crop)
    unpaired = TrainUnlabeled(dataroot=str(work), phase="unlabeled", finesize=args.crop)
    val = ValLabeled(dataroot=str(work), phase="val", finesize=args.crop)
    # num_workers is the only addition: it does not change the data. Pinned
    # memory is off and workers are few to keep host RAM down (a run was
    # SIGKILLed mid-epoch with no traceback, most likely a host OOM).
    kw = dict(num_workers=args.workers, pin_memory=False)
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

    # Upstream creates MUSIQ with as_loss=True and indexes the result per image.
    # In current pyiqa, as_loss=True averages over the batch (a 0-d tensor, so
    # upstream crashes) and keeps an autograd graph (OOM on 16 GB GPUs). All
    # call sites .detach() the scores and only compare them, so a per-image,
    # no-grad MUSIQ restores the intended behaviour. The range check is off
    # because raw network outputs can fall slightly outside [0, 1].
    import pyiqa
    del tr.iqa_metric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    musiq = pyiqa.create_metric("musiq", as_loss=False, check_input_range=False,
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tr.iqa_metric = lambda x: musiq(x).reshape(-1)

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

    cur = {"epoch": start}
    memlog.status = lambda: {"epoch": cur["epoch"], "iter": tr.curiter}

    epoch_times = []
    for epoch in range(start, args.epochs + 1):
        cur["epoch"] = epoch
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
        # bank before last.pt, so a checkpoint's bank is never older than it
        runstate.write_pack({"bank": bank}, run / "bank.pack")
        atomic_save({"student": tr.model.module.state_dict(), "teacher": tr.tmodel.state_dict(),
                     "optimizer": tr.optimizer_s.state_dict(), "scheduler": tr.lr_scheduler_s.state_dict(),
                     "epoch": epoch, "curiter": tr.curiter, "best_val_psnr": best, "rng": rng_state()},
                    last)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = memlog.log(event="epoch_end")
        rec.update({k: m.get(k) for k in ("rss_main", "rss_children", "cg_current", "cg_peak", "host_avail")})
        with open(run / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[epoch {epoch}] val PSNR {val_psnr:.3f} (best {best:.3f}) {dt:.0f}s", flush=True)

        elapsed = (time.time() - t_start) / 3600
        if epoch < args.epochs and elapsed + 1.3 * np.mean(epoch_times[-5:]) / 3600 > args.max_hours:
            print(f"time budget reached after epoch {epoch} ({elapsed:.2f}h); resume in a new session")
            memlog.log(event="budget_stop")
            return

    memlog.log(event="done")
    (run / "DONE").touch()
    print("training finished")


if __name__ == "__main__":
    main()
