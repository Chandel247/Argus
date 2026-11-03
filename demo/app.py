"""Baseline demo: upload an underwater image, see it enhanced.

Serves whatever exports/<model>/current.json points to and hot-reloads when
the exporter switches it, so a newly exported best checkpoint shows up
without restarting the demo or touching training.

.venv/bin/python demo/app.py --model semi_uir            # http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uie import baselines  # noqa: E402

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "static" / "examples"
MAX_UPLOAD = 15 * 1024 * 1024


class ModelSlot:
    """Holds the current enhancer; reloads when current.json changes."""

    def __init__(self, model: str, exports: Path, device: str):
        self.model, self.device = model, device
        self.pointer = exports / model / "current.json"
        self.lock = threading.Lock()
        self.mtime, self.meta, self.enhancer = None, None, None

    def get(self):
        with self.lock:
            if not self.pointer.exists():
                raise RuntimeError(f"no exported weights yet ({self.pointer}); run scripts/export_best.py")
            mtime = self.pointer.stat().st_mtime
            if mtime != self.mtime:
                meta = json.loads(self.pointer.read_text())
                weights = self.pointer.parent / meta["weights"]
                t0 = time.time()
                self.enhancer = baselines.get(self.model).Enhancer(weights, device=self.device)
                self.meta, self.mtime = meta, mtime
                print(f"loaded {self.model}/{meta['tag']} in {time.time() - t0:.1f}s", flush=True)
            return self.enhancer, self.meta


def to_data_url(arr: np.ndarray, fmt="JPEG") -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, quality=92)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def create_app(slot: ModelSlot, max_side: int) -> Flask:
    app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD

    @app.get("/")
    def index():
        return send_from_directory(HERE / "static", "index.html")

    @app.get("/api/model")
    def model_info():
        try:
            _, meta = slot.get()
            return jsonify({"ok": True, "name": slot.model, **meta})
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 503

    @app.get("/api/examples")
    def examples():
        files = sorted(p.name for p in EXAMPLES.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) \
            if EXAMPLES.exists() else []
        return jsonify(files)

    @app.post("/api/enhance")
    def enhance():
        if "image" in request.files:
            raw = request.files["image"].read()
        elif request.json and "example" in request.json:
            name = Path(request.json["example"]).name
            raw = (EXAMPLES / name).read_bytes()
        else:
            return jsonify({"ok": False, "error": "no image"}), 400
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        except Exception:  # noqa: BLE001
            return jsonify({"ok": False, "error": "not a readable image"}), 400
        try:
            enhancer, meta = slot.get()
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 503

        arr = np.asarray(img)
        t0 = time.time()
        out = enhancer(arr, size=None, max_side=max_side)
        ms = (time.time() - t0) * 1000
        # show input at the same resolution the model processed
        inp = np.asarray(img.resize((out.shape[1], out.shape[0]), Image.LANCZOS))
        return jsonify({"ok": True, "input": to_data_url(inp), "output": to_data_url(out),
                        "ms": round(ms), "size": [out.shape[1], out.shape[0]],
                        "orig_size": list(img.size), "checkpoint": meta["tag"]})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="semi_uir")
    ap.add_argument("--exports", type=Path, default=ROOT / "exports")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-side", type=int, default=512, help="longer side the model runs at")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    slot = ModelSlot(a.model, a.exports, a.device)
    try:
        slot.get()  # load eagerly so the first request is fast
    except RuntimeError as e:
        print(f"warning: {e}")
    create_app(slot, a.max_side).run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
