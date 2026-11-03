"""Run a baseline over a folder of images and save the enhanced outputs.

python scripts/infer.py --model semi_uir --weights exports/semi_uir/<tag>/weights.pt \
    --input data/uieb/raw-890 --split splits/uieb_test.txt --out results/semi_uir/uieb_t90/images
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uie import baselines  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--split", type=Path, help="names to process (default: all)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=256, help="square resize; 0 = keep aspect (see --max-side)")
    ap.add_argument("--max-side", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    enhance = baselines.get(args.model).Enhancer(args.weights, device=device)

    files = {p.stem: p for p in args.input.iterdir() if p.suffix.lower() in IMG_EXT}
    stems = ([Path(l.strip()).stem for l in args.split.read_text().splitlines() if l.strip()]
             if args.split else sorted(files))
    missing = [s for s in stems if s not in files]
    if missing:
        raise SystemExit(f"{len(missing)} inputs missing, e.g. {missing[:5]}")

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, s in enumerate(stems, 1):
        img = np.asarray(Image.open(files[s]).convert("RGB"))
        out = enhance(img, size=args.size or None, max_side=args.max_side)
        Image.fromarray(out).save(args.out / f"{s}.png")
        if i % 50 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)}  {(time.time() - t0) / i:.2f}s/img", file=sys.stderr)


if __name__ == "__main__":
    main()
