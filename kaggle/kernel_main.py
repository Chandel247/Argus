"""Entry point executed inside a Kaggle kernel. launch.py injects CONFIG.

Finds the attached code/data/state inputs under /kaggle/input by content (so
it does not depend on Kaggle's mount naming), then runs CONFIG["steps"]:
  check_official  official Semi-UIR weights on UIEB T90 (validates our shims)
  train           scripts/train_semi_uir.py, resuming from attached state
  test            best checkpoint on T90 / C60 / L400 + shared metrics
Everything worth keeping is written to /kaggle/working (the kernel output).
"""
CONFIG = {}  # replaced by launch.py

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT, OUT, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/argus")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_dir(marker: str, depth: int = 6) -> Path | None:
    """First directory under /kaggle/input containing relative path `marker`."""
    for root, dirs, _ in os.walk(INPUT):
        if Path(root, marker).exists():
            return Path(root)
        if len(Path(root).relative_to(INPUT).parts) >= depth:
            dirs[:] = []
    return None


def sh(cmd, **kw):
    log("$ " + " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    log(f"config: {json.dumps(CONFIG)}")
    code_src = find_dir("uie/metrics.py")
    assert code_src, "argus code dataset not attached"
    if TMP.exists():
        shutil.rmtree(TMP)
    shutil.copytree(code_src, TMP)
    os.chdir(TMP)
    py = sys.executable

    missing = [m for m in ("pyiqa", "adamp") if subprocess.run([py, "-c", f"import {m}"]).returncode]
    if missing:
        sh([py, "-m", "pip", "install", "-q", *missing])
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

    uieb = find_dir("raw-890")
    lsui = find_dir("input/0.jpg")  # LSUI/input + LSUI/GT
    assert uieb and lsui, f"datasets not found: uieb={uieb} lsui={lsui}"
    log(f"uieb={uieb} lsui={lsui}")
    run = OUT / "run"

    for step in CONFIG["steps"]:
        log(f"=== step: {step}")
        if step == "check_official":
            from_repo = TMP / "baselines/semi_uir/pretrained/model.pth"
            sh([py, "-c", "from uie.baselines import semi_uir; semi_uir.ensure_repo()"])
            pred = Path("/tmp/check/official_t90")
            sh([py, "scripts/infer.py", "--model", "semi_uir", "--weights", from_repo,
                "--input", uieb / "raw-890", "--split", "splits/uieb_test.txt", "--out", pred])
            sh([py, "scripts/evaluate.py", "--pred", pred, "--gt", uieb / "reference-890",
                "--split", "splits/uieb_test.txt", "--metrics", "psnr,ssim,uiqm,uciqe",
                "--name", "semi_uir_official", "--dataset", "uieb_t90", "--out", OUT / "results"])

        elif step == "train":
            state = find_dir("ckpt/last.pt")
            cmd = [py, "scripts/train_semi_uir.py", "--uieb", uieb, "--lsui", lsui,
                   "--run-dir", run, "--work-dir", "/tmp/work",
                   "--epochs", CONFIG.get("epochs", 200), "--max-hours", CONFIG.get("max_hours", 11)]
            if state:
                log(f"resuming from {state}")
                cmd += ["--resume-from", state]
            if CONFIG.get("smoke"):
                cmd += ["--smoke"]
            sh(cmd)

        elif step == "test":
            best = run / "ckpt/best.pt"
            if not best.exists() or (CONFIG.get("test_only_when_done", True) and not (run / "DONE").exists()):
                log("skipping test: training not finished")
                continue
            c60 = find_dir("challenging-60")
            sets = [("uieb_t90", uieb / "raw-890", "splits/uieb_test.txt", uieb / "reference-890"),
                    ("lsui_l400", lsui / "input", "splits/lsui_test.txt", lsui / "GT"),
                    ("uieb_c60", c60 / "challenging-60", None, None)]
            name = CONFIG.get("result_name", "semi_uir")
            for ds, inp, split, gt in sets:
                pred = Path(f"/tmp/pred/{ds}")
                cmd = [py, "scripts/infer.py", "--model", "semi_uir", "--weights", best, "--input", inp, "--out", pred]
                sh(cmd + (["--split", split] if split else []))
                ev = [py, "scripts/evaluate.py", "--pred", pred, "--name", name, "--dataset", ds,
                      "--out", OUT / "results"]
                if gt:
                    ev += ["--gt", gt, "--split", split, "--metrics", "psnr,ssim,lpips,uiqm,uciqe"]
                else:
                    ev += ["--metrics", "uiqm,uciqe,musiq,niqe,uranker"]
                sh(ev)
                # keep a few outputs for visual comparison
                keep = OUT / "results" / name / "samples" / ds
                keep.mkdir(parents=True, exist_ok=True)
                for p in sorted(pred.iterdir())[:12]:
                    shutil.copy2(p, keep / p.name)
        else:
            raise SystemExit(f"unknown step {step}")
    log("all steps done")


if __name__ == "__main__":
    main()
