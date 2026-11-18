"""RUOD (COCO annotations) -> Ultralytics YOLO dataset.

Layout written under `out`:
  images/<set>/<file>      a symlink to the original, or a resized copy
  labels/<set>/<stem>.txt  one "class cx cy w h" line per box, normalised to 0-1
  data.yaml                names + the set folders
Labels are normalised by the ORIGINAL width/height from the annotation file,
so they stay valid for any copy of the image that keeps the full frame at a
different size (a resized training copy, or an enhanced image from a UIE
method, even 256x256): only cropping or padding would break them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

JPEG_EXT = (".jpg", ".jpeg")


def load_coco(path: Path):
    """-> (class names in category-id order, {file_name: (w, h, [(cls, x, y, bw, bh), ...])})"""
    d = json.loads(Path(path).read_text())
    cats = sorted(d["categories"], key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}
    by_id = {i["id"]: (i["file_name"], i["width"], i["height"]) for i in d["images"]}
    images = {f: (w, h, []) for f, w, h in by_id.values()}
    for a in d["annotations"]:
        f = by_id[a["image_id"]][0]
        images[f][2].append((index[a["category_id"]], *a["bbox"]))
    return [c["name"] for c in cats], images


def yolo_lines(boxes, w: int, h: int) -> list[str]:
    """COCO [x, y, w, h] in pixels -> YOLO lines, clipped to the image.
    Boxes with no area left after clipping are dropped."""
    out = []
    for cls, x, y, bw, bh in boxes:
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        out.append(f"{cls} {cx:.6f} {cy:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return out


def _orientation(img) -> int:
    try:
        return int(img.getexif().get(0x0112, 1))
    except Exception:
        return 1


def _place(job):
    """Link or copy one image. Returns (status, file) where status is
    'link', 'copy', 'resized' or 'size_mismatch'."""
    src, dst, w, h, max_side = job
    from PIL import Image

    with Image.open(src) as im:
        raw = im.size
        rotated = _orientation(im) != 1
        big = max_side and max(raw) > max_side
        if not rotated and not big:
            if not dst.exists():
                dst.symlink_to(Path(src).resolve())
            return ("link" if raw == (w, h) else "size_mismatch"), dst.name
        # an EXIF rotation flag would make the loader turn the pixels while
        # the boxes stay in raw coordinates: write an un-flagged copy instead
        if big:
            s = max_side / max(raw)
            size = (max(1, round(raw[0] * s)), max(1, round(raw[1] * s)))
            im.draft("RGB", size)  # fast JPEG decode at reduced scale
            im = im.convert("RGB").resize(size, Image.LANCZOS)
        else:
            im = im.convert("RGB")
        tmp = dst.with_name(dst.name + ".tmp")
        im.save(tmp, "JPEG", quality=95)
        os.replace(tmp, dst)
    if raw != (w, h):
        return "size_mismatch", dst.name
    return ("resized" if big else "copy"), dst.name


def build_set(out: Path, name: str, src_dir: Path, files: list[str], images: dict,
              max_side: int | None = None, workers: int | None = None) -> dict:
    """Write images/<name>/ and labels/<name>/ for `files`. Returns counts."""
    img_dir, lab_dir = out / "images" / name, out / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in files if f not in images]
    if missing:
        raise KeyError(f"{len(missing)} files have no annotations, e.g. {missing[:3]}")
    n_boxes = 0
    for f in files:
        w, h, boxes = images[f]
        lines = yolo_lines(boxes, w, h)
        n_boxes += len(lines)
        (lab_dir / f"{Path(f).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    jobs = [(src_dir / f, img_dir / f, images[f][0], images[f][1], max_side) for f in files]
    stats = {"images": len(files), "boxes": n_boxes}
    if workers == 0:
        results = map(_place, jobs)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(max_workers=workers or os.cpu_count() or 2, mp_context=mp.get_context("spawn"))
        results = ex.map(_place, jobs, chunksize=16)
    mismatched = []
    for status, f in results:
        stats[status] = stats.get(status, 0) + 1
        if status == "size_mismatch":
            mismatched.append(f)
    if workers != 0:
        ex.shutdown()
    if mismatched:
        stats["size_mismatch_examples"] = mismatched[:5]
    return stats


def write_yaml(out: Path, names: list[str], sets: dict[str, str], filename: str = "data.yaml") -> Path:
    """sets: {'train': 'images/train', 'val': 'images/val', ...}. Plain YAML
    by hand (no PyYAML needed). Ultralytics requires both train and val keys,
    so an evaluation-only file points both at the set being scored."""
    lines = [f"path: {out.resolve()}"] + [f"{k}: {v}" for k, v in sets.items()] + ["names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    p = out / filename
    p.write_text("\n".join(lines) + "\n")
    return p


def read_split(path: Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
