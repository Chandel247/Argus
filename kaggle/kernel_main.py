"""Entry point executed inside a Kaggle kernel. launch.py injects CONFIG.

Finds the attached code/data/state inputs under /kaggle/input by content (so
it does not depend on Kaggle's mount naming), then runs CONFIG["steps"]:
  check_official  official Semi-UIR weights on UIEB T90 (validates our shims)
  train           scripts/train_semi_uir.py, resuming from attached state;
                  if the trainer is SIGKILLed (host OOM) it is restarted from
                  its own last checkpoint, up to CONFIG["kill_retries"] times
  test            best checkpoint on T90 / C60 / L400 + shared metrics
  ruod_prepare    RUOD -> YOLO dataset in /tmp/ruod_yolo (scripts/prepare_ruod_yolo.py)
  yolo_train      fine-tune YOLO on RUOD (scripts/train_yolo.py), resumable
  yolo_eval       best.pt on the RUOD evaluation sets (scripts/eval_detection.py)
  det_folders     the shared detector on bench400 image folders (raw + enhanced)
  cost            parameters, weight size, img/s of all models (scripts/bench_cost.py)
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


def check_state(state: Path | None):
    """Refuse to resume from an older state version than the one launch.py uploaded."""
    want = CONFIG.get("state_id")
    if not state or not want:
        return
    f = state / "STATE_ID"
    got = f.read_text().strip() if f.exists() else "?"
    log(f"state {got}")
    if got != want:
        raise SystemExit(f"attached state is {got}, expected {want}: Kaggle mounted an older version")


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


YOLO_DATA = Path("/tmp/ruod_yolo")
EVAL_SPLITS = {"test": "ruod_test.txt", "bench400": "ruod_bench400.txt", "blur": "ruod_env_blur.txt",
               "color": "ruod_env_color.txt", "light": "ruod_env_light.txt"}


def run_detection(py):
    """RUOD detection jobs (YOLO fine-tuning)."""
    ruod = find_dir("RUOD_pic/test")
    assert ruod, "RUOD dataset not attached"
    log(f"ruod={ruod}")
    sh([py, "-m", "pip", "install", "-q", CONFIG.get("ultralytics", "ultralytics==8.4.180")])
    sh([py, "-c", "import ultralytics, torch; print('ultralytics', ultralytics.__version__, 'torch', torch.__version__)"])
    run = OUT / "run"
    name = CONFIG.get("run_name", "yolo")
    weights = run / name / "weights"
    for step in CONFIG["steps"]:
        log(f"=== step: {step}")
        if step == "ruod_prepare":
            cmd = [py, "scripts/prepare_ruod_yolo.py", "--ruod", ruod, "--out", YOLO_DATA]
            if CONFIG.get("prepare_sets"):
                cmd += ["--sets", ",".join(CONFIG["prepare_sets"])]
            sh(cmd)
        elif step == "yolo_train":
            state = find_dir(f"{name}/weights/last.pt")
            check_state(state)
            if state and not (run / name / "weights" / "last.pt").exists():
                log(f"restoring state from {state}")
                shutil.copytree(state, run, dirs_exist_ok=True)
            if (run / "DONE").exists():
                log("training already finished")
                continue
            base = [py, "scripts/train_yolo.py", "--data", YOLO_DATA / "data.yaml", "--project", run,
                    "--name", name, "--model", CONFIG.get("model", "yolo11s.pt"),
                    "--epochs", CONFIG.get("epochs", 100), "--batch", CONFIG.get("batch", 32),
                    "--imgsz", CONFIG.get("imgsz", 640), "--fraction", CONFIG.get("fraction", 1.0)]
            retries = CONFIG.get("kill_retries", 1)
            while True:
                budget = CONFIG.get("max_hours", 10) - (time.time() - T0) / 3600
                cmd = base + ["--max-hours", round(budget, 2)]
                if (weights / "last.pt").exists():
                    cmd += ["--resume"]
                try:
                    sh(cmd)
                    break
                except subprocess.CalledProcessError as e:
                    memory_report(run)
                    if e.returncode != -9 or retries <= 0 or not (weights / "last.pt").exists() or budget < 0.5:
                        raise
                    retries -= 1
                    log(f"trainer was SIGKILLed; resuming from {weights / 'last.pt'}")
        elif step == "yolo_eval":
            best = weights / "best.pt"
            if not best.exists() or (CONFIG.get("eval_only_when_done", True) and not (run / "DONE").exists()):
                log("skipping eval: training not finished")
                continue
            for s in CONFIG.get("eval_sets", ["bench400", "test", "blur", "color", "light"]):
                sh([py, "scripts/eval_detection.py", "--weights", best, "--images", YOLO_DATA / "images" / s,
                    "--ann", ruod / "RUOD_ANN/instances_test.json", "--split", f"splits/{EVAL_SPLITS[s]}",
                    "--name", CONFIG.get("result_name", "yolo_ruod"), "--dataset", f"ruod_{s}",
                    "--out", OUT / "results"])
        elif step == "cost":
            gan, semi, yolo = find_dir("runs/gan/best.pt"), find_dir("ckpt/best.pt"), find_dir("yolo11s_ruod_best.pt")
            uieb = find_dir("raw-890")
            assert gan and semi and yolo and uieb, f"inputs missing: gan={gan} semi={semi} yolo={yolo} uieb={uieb}"
            sh([py, "scripts/bench_cost.py", "--gan", gan / "runs/gan/best.pt", "--semi", semi / "ckpt/best.pt",
                "--yolo", yolo / "yolo11s_ruod_best.pt", "--enh-images", uieb / "raw-890",
                "--enh-split", "splits/uieb_test.txt", "--det-images", ruod / "RUOD_pic/test",
                "--det-split", "splits/ruod_bench400.txt", "--out", OUT / "results/cost.json"])
        elif step == "det_folders":
            # one detector, one setting, several image folders of bench400
            best = find_dir("yolo11s_ruod_best.pt")
            assert best, "detector weights dataset not attached"
            for name, folder in CONFIG["det_folders"].items():
                if folder is None:
                    images = ruod / "RUOD_pic/test"
                else:
                    parent = find_dir(folder)
                    assert parent, f"{folder} not attached"
                    images = parent / folder
                n = sum(1 for _ in images.iterdir())
                log(f"{name}: {images} ({n} files)")
                sh([py, "scripts/eval_detection.py", "--weights", best / "yolo11s_ruod_best.pt", "--images", images,
                    "--ann", ruod / "RUOD_ANN/instances_test.json", "--split", "splits/ruod_bench400.txt",
                    "--name", name, "--dataset", "ruod_bench400", "--out", OUT / "results"])
        else:
            raise SystemExit(f"unknown step {step}")


def main():
    log(f"config: {json.dumps(CONFIG)}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # GIT_COMMIT marks our argus-code dataset; attached notebook outputs can
    # hold other clones of the repo (seen with a teammate's notebook)
    code_src = find_dir("GIT_COMMIT") or find_dir("uie/metrics.py")
    assert code_src, "argus code dataset not attached"
    if TMP.exists():
        shutil.rmtree(TMP)
    shutil.copytree(code_src, TMP)
    os.chdir(TMP)
    got = (TMP / "GIT_COMMIT").read_text().strip() if (TMP / "GIT_COMMIT").exists() else "?"
    want = CONFIG.get("code_commit")
    log(f"code at {got}")
    if want and got != want:
        raise SystemExit(f"attached argus-code is {got}, expected {want}: Kaggle mounted an older version")
    py = sys.executable

    missing = [m for m in ("pyiqa", "adamp") if subprocess.run([py, "-c", f"import {m}"]).returncode]
    if missing:
        sh([py, "-m", "pip", "install", "-q", *missing])
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

    run = OUT / "run"
    if any(st in CONFIG["steps"] for st in ("ruod_prepare", "yolo_train", "yolo_eval", "det_folders", "cost")):
        run_detection(py)
        log("all steps done")
        return
    uieb = find_dir("raw-890")
    lsui = find_dir("input/0.jpg")  # LSUI/input + LSUI/GT
    assert uieb and lsui, f"datasets not found: uieb={uieb} lsui={lsui}"
    log(f"uieb={uieb} lsui={lsui}")

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
            check_state(state)
            epochs = CONFIG.get("epochs", 200) + (1 if step == "train_resume" else 0)
            cmd = [py, "scripts/train_semi_uir.py", "--uieb", uieb, "--lsui", lsui,
                   "--run-dir", run, "--work-dir", "/tmp/work",
                   "--epochs", epochs, "--max-hours", CONFIG.get("max_hours", 11)]
            if state:
                log(f"resuming from {state}")
                cmd += ["--resume-from", state]
            if CONFIG.get("smoke"):
                cmd += ["--smoke"]
            if CONFIG.get("restart_every"):
                cmd += ["--restart-every", CONFIG["restart_every"]]
            retries = CONFIG.get("kill_retries", 2)
            while True:
                try:
                    sh(cmd)
                    break
                except subprocess.CalledProcessError as e:
                    planned = e.returncode == 3  # trainer asked for a fresh process (--restart-every)
                    if not planned:
                        memory_report(run)
                    if not planned and (e.returncode != -9 or retries <= 0) or not (run / "ckpt/last.pt").exists():
                        raise
                    if planned:
                        log("planned trainer restart (releases host memory)")
                    else:
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
                        if planned:  # checkpoint is saved; end the session normally
                            break
                        raise

        elif step == "test":
            best = run / "ckpt/best.pt"
            if not best.exists() and "train" not in CONFIG["steps"]:
                state = find_dir("ckpt/best.pt")  # test-only job: checkpoint comes as attached state
                check_state(state)
                if state:
                    best = state / "ckpt/best.pt"
                    log(f"testing {best}")
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
