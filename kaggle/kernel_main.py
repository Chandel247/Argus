"""Entry point executed inside a Kaggle kernel. launch.py injects CONFIG.

Finds the attached code/data/state inputs under /kaggle/input by content (so
it does not depend on Kaggle's mount naming), then runs CONFIG["steps"]:
  check_official  official Semi-UIR weights on UIEB T90 (validates our shims)
  train           scripts/train_semi_uir.py, resuming from attached state;
                  if the trainer is SIGKILLed (host OOM) it is restarted from
                  its own last checkpoint, up to CONFIG["kill_retries"] times
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
T0 = time.time()


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


def memory_report(run: Path):
    """After a failure: container OOM-kill counter and the last telemetry lines."""
    try:
        sys.path.insert(0, str(TMP))
        from uie.runstate import cgroup_memory, gb
        m = cgroup_memory()
        log(f"cgroup: oom_kills={m['oom_kills']} current={gb(m['cg_current'])}GB "
            f"peak={gb(m['cg_peak'])}GB max={gb(m['cg_max'])}GB")
    except Exception as e:
        log(f"cgroup read failed: {e}")
    mem = run / "mem.log"
    if mem.exists():
        for line in mem.read_text().splitlines()[-5:]:
            log(f"mem.log: {line}")


def main():
    log(f"config: {json.dumps(CONFIG)}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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

        elif step in ("train", "train_resume"):
            # train_resume: run the trainer again for one more epoch; it must
            # continue from run/ckpt/last.pt (tests resume within one session)
            state = find_dir("ckpt/last.pt") if step == "train" else None
            epochs = CONFIG.get("epochs", 200) + (1 if step == "train_resume" else 0)
            cmd = [py, "scripts/train_semi_uir.py", "--uieb", uieb, "--lsui", lsui,
                   "--run-dir", run, "--work-dir", "/tmp/work",
                   "--epochs", epochs, "--max-hours", CONFIG.get("max_hours", 11)]
            if state:
                log(f"resuming from {state}")
                cmd += ["--resume-from", state]
            if CONFIG.get("smoke"):
                cmd += ["--smoke"]
            retries = CONFIG.get("kill_retries", 2)
            while True:
                try:
                    sh(cmd)
                    break
                except subprocess.CalledProcessError as e:
                    memory_report(run)
                    if e.returncode != -9 or retries <= 0 or not (run / "ckpt/last.pt").exists():
                        raise
                    retries -= 1
                    log(f"trainer was SIGKILLed; restarting from {run}/ckpt/last.pt ({retries} retries left)")
                    # state is now in the run dir; --max-hours counts from the
                    # restart, so shrink it by the time already used
                    used = (time.time() - T0) / 3600
                    cmd = [c for c in cmd]
                    if "--resume-from" in cmd:
                        i = cmd.index("--resume-from")
                        del cmd[i:i + 2]
                    i = cmd.index("--max-hours")
                    cmd[i + 1] = round(CONFIG.get("max_hours", 11) - used, 2)
                    if cmd[i + 1] < 0.5:
                        log("not enough session time left to restart")
                        raise

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
