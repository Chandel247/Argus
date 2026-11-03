"""Score a folder of enhanced images with the shared benchmark metrics.

Pairs predictions with references by filename stem, so a method that writes
.png outputs for .jpg inputs still matches.

Examples
--------
# UIEB T90, full-reference + no-reference
python scripts/evaluate.py --pred results/ucolor/uieb_t90 \
    --gt data/uieb/reference-890 --split splits/uieb_test.txt \
    --name ucolor --dataset uieb_t90

# Score the raw inputs themselves (the "Input" row of the table)
python scripts/evaluate.py --pred data/uieb/raw-890 --gt data/uieb/reference-890 \
    --split splits/uieb_test.txt --name input --dataset uieb_t90

# No-reference only (C60, RUOD)
python scripts/evaluate.py --pred results/ucolor/c60 --name ucolor --dataset uieb_c60
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uie import metrics as M  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASSIC = {"psnr": M.psnr, "ssim": M.ssim, "uiqm": M.uiqm, "uciqe": M.uciqe}


def index_dir(d: Path) -> dict[str, Path]:
    files = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
    idx = {p.stem: p for p in files}
    if len(idx) != len(files):
        raise SystemExit(f"duplicate filename stems in {d}")
    return idx


def load(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size and im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path, help="folder of enhanced images")
    ap.add_argument("--gt", type=Path, help="folder of reference images (enables full-reference metrics)")
    ap.add_argument("--split", type=Path, help="file of image names to evaluate (default: all in --pred)")
    ap.add_argument("--metrics", help="comma list; default: psnr,ssim,lpips (if --gt) + uiqm,uciqe")
    ap.add_argument("--size", type=int, default=256, help="resize both images to size x size (0 = native)")
    ap.add_argument("--name", required=True, help="method name, e.g. ucolor")
    ap.add_argument("--dataset", required=True, help="test set name, e.g. uieb_t90")
    ap.add_argument("--out", type=Path, default=ROOT / "results", help="output root")
    ap.add_argument("--device", help="torch device for deep metrics (default: auto)")
    args = ap.parse_args(argv)

    if args.metrics:
        names = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    else:
        names = (["psnr", "ssim", "lpips"] if args.gt else []) + ["uiqm", "uciqe"]
    unknown = set(names) - set(M.HIGHER_IS_BETTER)
    if unknown:
        raise SystemExit(f"unknown metrics: {sorted(unknown)}")
    fr = [n for n in names if n in M.FULL_REFERENCE]
    if fr and not args.gt:
        raise SystemExit(f"{fr} need --gt")

    preds = index_dir(args.pred)
    gts = index_dir(args.gt) if args.gt else {}
    if args.split:
        stems = [Path(l.strip()).stem for l in args.split.read_text().splitlines() if l.strip()]
    else:
        stems = sorted(preds)
    missing = [s for s in stems if s not in preds] + [s for s in stems if gts and s not in gts]
    if missing:
        raise SystemExit(f"{len(missing)} images missing from pred/gt, e.g. {missing[:5]}")

    deep_names = [n for n in names if n in M.DEEP]
    deep = M.DeepMetrics(deep_names, args.device) if deep_names else None

    rows = []
    for i, s in enumerate(stems, 1):
        p = load(preds[s], args.size)
        g = load(gts[s], args.size) if gts else None
        row = {"image": s}
        for n in names:
            if n in CLASSIC:
                row[n] = CLASSIC[n](p, g) if n in M.FULL_REFERENCE else CLASSIC[n](p)
        if deep:
            row.update(deep(p, g))
        rows.append(row)
        if i % 50 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)}", file=sys.stderr)

    out_dir = args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.dataset}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", *names])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "method": args.name, "dataset": args.dataset, "n": len(rows), "size": args.size,
        "mean": {n: float(np.mean([r[n] for r in rows])) for n in names},
    }
    (out_dir / f"{args.dataset}.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{args.name} on {args.dataset} (n={len(rows)}, size={args.size or 'native'})")
    for n in names:
        arrow = "↑" if M.HIGHER_IS_BETTER[n] else "↓"
        print(f"  {n:8s}{arrow} {summary['mean'][n]:.4f}")
    return summary


if __name__ == "__main__":
    main()
