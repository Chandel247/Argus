"""Score a folder of RUOD images with a fine-tuned YOLO detector.

This is the downstream test for enhancement methods: enhance the benchmark
images, then run this on the enhanced folder and on the originals, and
compare mAP. Images are matched to annotations by file stem, so an enhancer
may change the extension (000123.jpg -> 000123.png). Boxes are normalised by
the original image size, so outputs at another resolution (e.g. 256x256) are
scored correctly as long as the full frame is kept (no cropping or padding);
note the detector then sees less detail, which lowers mAP by itself.

python scripts/eval_detection.py --weights best.pt --images <folder> \\
    --ann RUOD_ANN/instances_test.json --split splits/ruod_bench400.txt \\
    --name my_method --dataset ruod_bench400 --out results

Writes <out>/<name>/<dataset>_det.json: mAP50, mAP50-95, precision, recall,
per-class AP, and how many images were found at a different size.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uie import ruod  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def build(images_dir: Path, files: list[str], coco_images: dict, names: list[str], work: Path) -> dict:
    by_stem = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    missing = [f for f in files if Path(f).stem not in by_stem]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(files)} images not found in {images_dir}, e.g. {missing[:3]}")
    from PIL import Image
    img_dir, lab_dir = work / "images" / "eval", work / "labels" / "eval"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    resized = 0
    for f in files:
        src = by_stem[Path(f).stem]
        w, h, boxes = coco_images[f]
        with Image.open(src) as im:
            resized += im.size != (w, h)
        (img_dir / src.name).symlink_to(src.resolve())
        lines = ruod.yolo_lines(boxes, w, h)
        (lab_dir / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    ruod.write_yaml(work, names, {"train": "images/eval", "val": "images/eval"})
    return {"images": len(files), "resized": resized}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="folder with the (enhanced) images")
    ap.add_argument("--ann", type=Path, required=True, help="RUOD_ANN/instances_test.json")
    ap.add_argument("--split", type=Path, default=ROOT / "splits" / "ruod_bench400.txt")
    ap.add_argument("--name", required=True, help="method name, e.g. raw or semi_uir")
    ap.add_argument("--dataset", default="ruod_bench400")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu (default: GPU if available)")
    a = ap.parse_args(argv)

    names, coco_images = ruod.load_coco(a.ann)
    files = ruod.read_split(a.split)
    from scripts.train_yolo import quiet_integrations
    quiet_integrations()
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        info = build(a.images, files, coco_images, names, work)
        t0 = time.time()
        m = YOLO(str(a.weights)).val(data=str(work / "data.yaml"), split="val", imgsz=a.imgsz, batch=a.batch,
                                     device=a.device, project=str(work / "runs"), name="val", plots=False,
                                     verbose=False)
        box = m.box
        per_class = {names[int(c)]: {"ap50": round(float(box.ap50[i]), 4), "ap50_95": round(float(box.ap[i]), 4)}
                     for i, c in enumerate(box.ap_class_index)}
        res = {"method": a.name, "dataset": a.dataset, "weights": str(a.weights), "images_dir": str(a.images),
               "map50": round(float(box.map50), 4), "map50_95": round(float(box.map), 4),
               "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
               "per_class": per_class, **info, "imgsz": a.imgsz, "sec": round(time.time() - t0, 1)}
    out = a.out / a.name / f"{a.dataset}_det.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"{a.name} on {a.dataset}: mAP50 {res['map50']:.4f}  mAP50-95 {res['map50_95']:.4f} -> {out}")
    return res


if __name__ == "__main__":
    main()
