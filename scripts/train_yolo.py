"""Fine-tune an Ultralytics YOLO detector on the YOLO version of RUOD.

Starts from COCO-pretrained weights (e.g. yolo11s.pt) and trains on
data.yaml's train set, picking best.pt on its val set (held out of RUOD
train; the test set is never used for training decisions).

Resumable across Kaggle sessions: Ultralytics saves <project>/<name>/weights/
last.pt every epoch, and --resume continues from it (optimizer, schedule and
epoch included). --max-hours stops cleanly: once the budget is spent, the
watchdog waits for the next last.pt to be written, then stops training, so the
next session resumes from a complete epoch. (Ultralytics' own callbacks cannot
do this under multi-GPU training: its DDP workers are rebuilt from a
generated script that drops custom callbacks.)

python scripts/train_yolo.py --data /tmp/ruod_yolo/data.yaml --model yolo11s.pt \\
    --epochs 100 --project runs/yolo --name ruod --devices 0,1
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def quiet_integrations():
    """Kaggle images ship wandb/comet/mlflow; Ultralytics would try to log to
    them (wandb can block waiting for a login)."""
    for k in ("WANDB_MODE", "COMET_MODE"):
        os.environ[k] = "disabled"
    from ultralytics import settings
    for key in ("wandb", "comet", "clearml", "mlflow", "neptune", "dvc", "raytune", "hub", "tensorboard"):
        try:
            settings.update({key: False})
        except Exception:  # key not present in this version
            pass


def train(a):
    quiet_integrations()
    from ultralytics import YOLO

    last = a.project / a.name / "weights" / "last.pt"
    if a.resume and last.exists():
        print(f"resuming from {last}", flush=True)
        YOLO(str(last)).train(resume=True)
        return
    devices = [int(d) for d in a.devices.split(",")] if a.devices not in ("cpu", "") else "cpu"
    YOLO(a.model).train(
        data=str(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=devices,
        project=str(a.project), name=a.name, exist_ok=True, seed=0, workers=a.workers,
        fraction=a.fraction, patience=a.patience, plots=True, cache=False,
    )


def watch(a, argv):
    """Run training in a child process group; stop it after the first epoch
    that ends past the time budget."""
    from uie.runstate import MemLogger

    a.project.mkdir(parents=True, exist_ok=True)
    mem = MemLogger(a.project / "mem.log", 60)
    mem.log(event="start")
    mem.start()
    t0 = time.time()
    child = subprocess.Popen([sys.executable, __file__, *argv, "--child"], start_new_session=True)
    last = a.project / a.name / "weights" / "last.pt"
    seen = last.stat().st_mtime if last.exists() else 0.0
    stopped = False
    while child.poll() is None:
        time.sleep(15)
        mtime = last.stat().st_mtime if last.exists() else 0.0
        over = (time.time() - t0) / 3600 > a.max_hours
        if over and mtime > seen:
            time.sleep(30)  # let best.pt / results.csv for this epoch finish writing
            print(f"time budget of {a.max_hours} h spent after a saved epoch; stopping to resume later", flush=True)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            stopped = True
            break
        seen = mtime if not over else seen
    mem.log(event="budget_stop" if stopped else "exit", returncode=child.returncode)
    if stopped:
        return 0
    if child.returncode == 0:
        (a.project / "DONE").touch()
    return child.returncode


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32, help="total over all GPUs")
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--workers", type=int, default=4, help="per GPU, capped by the CPU count")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the train set (smoke tests)")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--name", default="yolo")
    ap.add_argument("--resume", action="store_true", help="continue from <project>/<name>/weights/last.pt")
    ap.add_argument("--max-hours", type=float, help="stop cleanly after the first epoch past this")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    a.project = a.project.resolve()
    if a.child or not a.max_hours:
        train(a)
        if not a.child:
            (a.project / "DONE").touch()
        return 0
    return watch(a, argv)


if __name__ == "__main__":
    sys.exit(main())
