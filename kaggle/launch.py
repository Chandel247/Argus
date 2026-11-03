"""Drive Kaggle runs from the local machine (stdlib only; uses the `kaggle` CLI).

  python kaggle/launch.py sync-code              upload git HEAD as private dataset <user>/argus-code
  python kaggle/launch.py push <job> [--resume]  start a session (--resume attaches saved state)
  python kaggle/launch.py status <job>
  python kaggle/launch.py logs <job>
  python kaggle/launch.py fetch <job>            download finished session output to runs/<job>/<stamp>
  python kaggle/launch.py save-state <job>       upload latest fetched state as <user>/argus-state-<job>
  python kaggle/launch.py next <job>             fetch + save-state + push --resume (chain sessions)

Long training is a chain of sessions: each stops cleanly before Kaggle's
limit, its output (checkpoints + reliable bank) is fetched, re-uploaded as a
private state dataset, and attached to the next session.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATASETS = {
    "uieb": "naomirory2/uieb-final",
    "lsui": "noureldin199/lsui-large-scale-underwater-image-dataset",
}

JOBS = {
    # official weights through our shims (passed: T90 24.12 dB) is step
    # "check_official"; this job smoke-tests training + resume (~10 min)
    "semi-uir-check": {"title": "argus semi-uir check", "steps": ["train", "train_resume"],
                       "epochs": 2, "smoke": True, "max_hours": 1, "timeout": 3600},
    # protocol run A: train UIEB, test T90/C60/L400
    "semi-uir-uieb": {"title": "argus semi-uir uieb", "steps": ["train", "test"],
                      "epochs": 200, "max_hours": 11, "result_name": "semi_uir"},
}
ACCELERATOR = "NvidiaTeslaT4"  # 2x T4; P100 (sm_60) is unsupported by current PyTorch builds


def kaggle(*args, capture=False, check=True):
    cmd = ["kaggle", *map(str, args)]
    print("$ " + " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if check and r.returncode:
        if capture:
            print(r.stdout, r.stderr, file=sys.stderr)
        raise SystemExit(f"kaggle command failed ({r.returncode})")
    return r.stdout if capture else None


def username():
    for name in ("credentials.json", "kaggle.json"):
        p = Path.home() / ".kaggle" / name
        if p.exists():
            return json.loads(p.read_text())["username"]
    raise SystemExit("no Kaggle credentials found in ~/.kaggle")


def wait_ready(ref, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = subprocess.run(["kaggle", "datasets", "status", ref], capture_output=True, text=True).stdout
        if "ready" in out.lower():
            return
        time.sleep(15)
    raise SystemExit(f"dataset {ref} not ready after {timeout}s")


def upload_dataset(ref, folder: Path, title, message):
    meta = {"title": title, "id": ref, "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta))
    exists = subprocess.run(["kaggle", "datasets", "status", ref], capture_output=True, text=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", folder, "-m", message, "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", folder, "--dir-mode", "zip")  # private by default
    wait_ready(ref)


def cmd_sync_code(_):
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("warning: uncommitted changes are NOT uploaded (git HEAD only)", file=sys.stderr)
    with tempfile.TemporaryDirectory() as d:
        code = Path(d) / "code"
        code.mkdir()
        archive = Path(d) / "head.tar"
        subprocess.run(["git", "-C", ROOT, "archive", "-o", archive, "HEAD"], check=True)
        subprocess.run(["tar", "-xf", archive, "-C", code], check=True)
        (code / "GIT_COMMIT").write_text(head + "\n")
        upload_dataset(f"{username()}/argus-code", code, "argus-code", f"code @ {head}")
    print(f"code synced at {head}")


def kernel_slug(job):
    return f"{username()}/argus-{job}"


def cmd_push(a):
    job = JOBS[a.job]
    cfg = {k: v for k, v in job.items() if k not in ("title", "timeout")}
    src = (ROOT / "kaggle" / "kernel_main.py").read_text()
    assert "CONFIG = {}" in src
    code = src.replace("CONFIG = {}", "CONFIG = " + repr(cfg), 1)  # Python literal, not JSON
    sources = [f"{username()}/argus-code", *DATASETS.values()]
    if a.resume:
        sources.append(f"{username()}/argus-state-{a.job}")
    meta = {"id": kernel_slug(a.job), "title": job["title"], "code_file": "main.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": ACCELERATOR, "dataset_sources": sources, "kernel_sources": [],
            "competition_sources": [], "model_sources": []}
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "main.py").write_text(code)
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        args = ["kernels", "push", "-p", d, "--accelerator", ACCELERATOR]
        if job.get("timeout"):
            args += ["-t", job["timeout"]]
        kaggle(*args)


def cmd_status(a):
    kaggle("kernels", "status", kernel_slug(a.job))


def cmd_logs(a):
    kaggle("kernels", "logs", kernel_slug(a.job), check=False)


def latest_fetch(job) -> Path:
    d = RUNS / job
    stamps = sorted(p for p in d.iterdir() if p.is_dir() and p.name[0].isdigit()) if d.exists() else []
    if not stamps:
        raise SystemExit(f"nothing fetched for {job}; run: launch.py fetch {job}")
    return stamps[-1]


def cmd_fetch(a):
    dest = RUNS / a.job / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True)
    kaggle("kernels", "output", kernel_slug(a.job), "-p", dest)
    print(f"fetched to {dest}")
    run = dest / "run"
    if (run / "metrics.jsonl").exists():
        lines = (run / "metrics.jsonl").read_text().splitlines()
        print(f"{len(lines)} epochs logged; last: {lines[-1] if lines else '-'}")
    return dest


def cmd_save_state(a):
    run = latest_fetch(a.job) / "run"
    if not (run / "ckpt" / "last.pt").exists():
        raise SystemExit(f"no ckpt/last.pt in {run}")
    with tempfile.TemporaryDirectory() as d:
        state = Path(d) / "state"
        shutil.copytree(run, state, ignore=shutil.ignore_patterns("_work"))
        upload_dataset(f"{username()}/argus-state-{a.job}", state, f"argus-state-{a.job}",
                       f"state from {run.parent.name}")


def cmd_next(a):
    cmd_fetch(a)
    cmd_save_state(a)
    a.resume = True
    cmd_push(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync-code").set_defaults(fn=cmd_sync_code)
    for name, fn in [("push", cmd_push), ("status", cmd_status), ("logs", cmd_logs), ("fetch", cmd_fetch),
                     ("save-state", cmd_save_state), ("next", cmd_next)]:
        p = sub.add_parser(name)
        p.add_argument("job", choices=sorted(JOBS))
        if name == "push":
            p.add_argument("--resume", action="store_true")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
