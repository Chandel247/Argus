"""Pick the best checkpoint and export its weights for the demo.

Read-only with respect to training: it only reads ckpt/best.pt and
metrics.jsonl from run directories (the trainer writes both atomically), so
it can run at any time, alongside a live run or on fetched Kaggle sessions.

Candidates are every run dir given (and, with --kaggle-job, every fetched
session under runs/<job>/). The one with the highest validation PSNR wins.
Its weights are written to exports/<model>/<tag>/ and exports/<model>/current.json
is switched to it atomically, which the demo watches and hot-reloads.
An export never replaces a better current one unless --force.

python scripts/export_best.py --model semi_uir --kaggle-job semi-uir-uieb
python scripts/export_best.py --model semi_uir --run runs/local_run --watch 300
python scripts/export_best.py --model semi_uir --official baselines/semi_uir/pretrained/model.pth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def state_dict_of(ckpt):
    if isinstance(ckpt, dict):
        for key in ("student", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    return {k.removeprefix("module."): v for k, v in ckpt.items()}


def candidates_from_run(run: Path):
    best = run / "ckpt" / "best.pt"
    if not best.exists():
        return []
    ck = load(best)
    session = run.parent.name if run.name == "run" else run.name
    return [{"path": best, "score": float(ck["val_psnr"]), "epoch": int(ck["epoch"]),
             "source": "trained", "run": str(run.relative_to(ROOT) if run.is_relative_to(ROOT) else run),
             "tag": f"{session}-e{int(ck['epoch']):03d}"}]


def atomic_write_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def export(model: str, cand: dict, out_root: Path, force: bool) -> bool:
    import torch
    mdir = out_root / model
    mdir.mkdir(parents=True, exist_ok=True)
    current_file = mdir / "current.json"
    current = json.loads(current_file.read_text()) if current_file.exists() else None

    if current and not force:
        cur_score = current.get("val_psnr")
        if current.get("tag") == cand["tag"]:
            print(f"already exported: {cand['tag']}")
            return False
        if cur_score is not None and (cand["score"] is None or cand["score"] <= cur_score):
            new = "unscored" if cand["score"] is None else f"{cand['score']:.3f}"
            print(f"keeping current {current['tag']} (val PSNR {cur_score:.3f}; candidate {new})")
            return False

    sd = state_dict_of(load(cand["path"]))
    final = mdir / cand["tag"]
    tmp = Path(tempfile.mkdtemp(dir=mdir, prefix=".tmp-"))
    try:
        torch.save(sd, tmp / "weights.pt")
        digest = hashlib.sha256((tmp / "weights.pt").read_bytes()).hexdigest()
        meta = {"model": model, "tag": cand["tag"], "source": cand["source"], "epoch": cand["epoch"],
                "val_psnr": cand["score"], "val_set": "UIEB val (50 held out of train)" if cand["score"] else None,
                "from": cand.get("run") or str(cand["path"]), "sha256": digest,
                "n_params": int(sum(v.numel() for v in sd.values())),
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    atomic_write_text(current_file, json.dumps({**meta, "weights": f"{cand['tag']}/weights.pt"}, indent=2) + "\n")
    score = f"val PSNR {cand['score']:.3f}" if cand["score"] is not None else "no val score"
    print(f"exported {model}/{cand['tag']} ({score}) -> {final}")
    return True


def fetch_kaggle(job: str):
    """Fetch the newest *finished* session; never touches a running one.
    Sessions that ended in an error (e.g. the trainer was killed) count as
    finished: checkpoints are written atomically and the fetch skips *.tmp."""
    r = subprocess.run([sys.executable, str(ROOT / "kaggle" / "launch.py"), "status", job],
                       capture_output=True, text=True)
    status = (r.stdout + r.stderr).lower()
    if not any(w in status for w in ("complete", "error", "cancel")):
        print(f"{job}: latest session not complete ({status.strip().splitlines()[-1] if status.strip() else '?'}); "
              "using sessions already fetched")
        return
    runs = ROOT / "runs" / job
    fetched = sorted(runs.iterdir()) if runs.exists() else []
    subprocess.run([sys.executable, str(ROOT / "kaggle" / "launch.py"), "fetch", job], check=True)
    if fetched:  # drop the new fetch if it is identical to the previous one
        new = sorted(runs.iterdir())[-1]
        a, b = fetched[-1] / "run/metrics.jsonl", new / "run/metrics.jsonl"
        if a.exists() and b.exists() and a.read_bytes() == b.read_bytes():
            shutil.rmtree(new)


def run_once(a) -> bool:
    cands = []
    if a.kaggle_job:
        if not a.no_fetch:
            fetch_kaggle(a.kaggle_job)
        jd = ROOT / "runs" / a.kaggle_job
        for s in sorted(jd.iterdir()) if jd.exists() else []:
            cands += candidates_from_run(s / "run")
    for r in a.run or []:
        cands += candidates_from_run(r)
    if a.official:
        cands.append({"path": a.official, "score": None, "epoch": None, "source": "official",
                      "tag": "official"})
    if not cands:
        print("no checkpoints found yet")
        return False
    trained = [c for c in cands if c["score"] is not None]
    best = max(trained, key=lambda c: c["score"]) if trained else cands[0]
    return export(a.model, best, a.out, a.force)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", type=Path, nargs="*", help="run dirs containing ckpt/best.pt")
    ap.add_argument("--kaggle-job", help="also consider all fetched sessions of this Kaggle job")
    ap.add_argument("--no-fetch", action="store_true", help="with --kaggle-job: do not fetch")
    ap.add_argument("--official", type=Path, help="export official released weights (no val score)")
    ap.add_argument("--out", type=Path, default=ROOT / "exports")
    ap.add_argument("--force", action="store_true", help="export even if not better than current")
    ap.add_argument("--watch", type=int, metavar="SEC", help="repeat every SEC seconds")
    a = ap.parse_args(argv)
    if a.official:
        a.force = a.force or not (a.out / a.model / "current.json").exists()
    while True:
        run_once(a)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
