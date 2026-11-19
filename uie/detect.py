"""Underwater object detection for the demo: image in -> boxed image + detections.

    from uie.detect import Detector
    det = Detector("exports/yolo11s_ruod/best.pt")      # load once
    boxed, detections = det.detect("photo.jpg")          # path, PIL image or RGB array
    # boxed: H x W x 3 uint8 RGB array, same size as the input
    # detections: [{"box": [x1, y1, x2, y2], "class": "fish", "confidence": 0.91}, ...]
    #             box in pixels of the input image, sorted by confidence

The detector is YOLO11s fine-tuned on RUOD (10 classes), the same weights
and settings (640 px) as the bench400 results in docs/ruod_benchmark.md.
Needs `pip install ultralytics==8.4.180`.

    python -m uie.detect --weights best.pt --image photo.jpg --out boxed.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# one fixed colour per RUOD class (index = class id), readable on blue-green water
PALETTE = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
           (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187)]


def load_rgb(image) -> np.ndarray:
    """Path / PIL image / array -> H x W x 3 uint8 RGB, EXIF rotation applied."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)
    im = image if isinstance(image, Image.Image) else Image.open(image)
    return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def to_detections(xyxy, cls, conf, names, width, height) -> list[dict]:
    """Raw model outputs -> list of {box, class, confidence}, clipped to the image."""
    out = []
    for (x1, y1, x2, y2), c, p in zip(np.asarray(xyxy, float).reshape(-1, 4), np.asarray(cls).reshape(-1),
                                      np.asarray(conf, float).reshape(-1)):
        box = [round(float(np.clip(x1, 0, width)), 1), round(float(np.clip(y1, 0, height)), 1),
               round(float(np.clip(x2, 0, width)), 1), round(float(np.clip(y2, 0, height)), 1)]
        out.append({"box": box, "class": names[int(c)], "class_id": int(c), "confidence": round(float(p), 4)})
    return sorted(out, key=lambda d: -d["confidence"])


def draw(rgb: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Copy of the image with a box and 'class 0.91' label per detection."""
    im = Image.fromarray(rgb).copy()
    d = ImageDraw.Draw(im)
    scale = max(im.size) / 1000
    width = max(2, round(3 * scale))
    try:
        font = ImageFont.load_default(size=max(12, round(18 * scale)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for det in sorted(detections, key=lambda x: x["confidence"]):  # most confident drawn last (on top)
        x1, y1, x2, y2 = det["box"]
        colour = PALETTE[det.get("class_id", 0) % len(PALETTE)]
        d.rectangle([x1, y1, x2, y2], outline=colour, width=width)
        label = f"{det['class']} {det['confidence']:.2f}"
        l, t, r, b = d.textbbox((0, 0), label, font=font)
        tw, th = (r - l) + 2 * width, (b - t) + 2 * width
        ty = y1 - th if y1 - th >= 0 else y1  # above the box, or inside it at the top edge
        tx = max(0, min(x1, im.width - tw))  # keep the label inside the image
        d.rectangle([tx, ty, tx + tw, ty + th], fill=colour)
        d.text((tx + width, ty + width - t), label, fill=(0, 0, 0), font=font)
    return np.asarray(im)


class Detector:
    """Loads the YOLO weights once; detect() is safe to call per request."""

    def __init__(self, weights: str | Path, device: str | None = None, imgsz: int = 640, conf: float = 0.25,
                 iou: float = 0.7):
        from ultralytics import YOLO  # imported here so the rest of the package works without it
        self.model = YOLO(str(weights))
        self.names = self.model.names  # {id: name}
        self.device, self.imgsz, self.conf, self.iou = device, imgsz, conf, iou

    def detect(self, image, conf: float | None = None) -> tuple[np.ndarray, list[dict]]:
        rgb = load_rgb(image)
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # ultralytics treats arrays as BGR (OpenCV order)
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf if conf is None else conf, iou=self.iou,
                               device=self.device, verbose=False)[0]
        b = r.boxes
        dets = to_detections(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy(), self.names,
                             rgb.shape[1], rgb.shape[0])
        return draw(rgb, dets), dets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", help="where to save the boxed image (default: print detections only)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="cpu, mps, 0 ... (default: auto)")
    a = ap.parse_args()
    boxed, dets = Detector(a.weights, device=a.device, conf=a.conf).detect(a.image)
    if a.out:
        Image.fromarray(boxed).save(a.out, quality=92)
    print(json.dumps(dets, indent=1))


if __name__ == "__main__":
    main()
