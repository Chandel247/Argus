import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from uie.detect import draw, load_rgb, to_detections

NAMES = {0: "holothurian", 1: "echinus", 4: "fish"}
WEIGHTS = Path(__file__).resolve().parent.parent / "exports/yolo11s_ruod/best.pt"


def test_to_detections_clips_sorts_and_names():
    dets = to_detections([[-5, 10, 50, 60], [10, 10, 300, 90]], [4, 1], [0.4, 0.9], NAMES, width=200, height=100)
    assert [d["class"] for d in dets] == ["echinus", "fish"]  # by confidence
    assert dets[0]["box"] == [10.0, 10.0, 200.0, 90.0]  # clipped to the image
    assert dets[1]["box"][0] == 0.0
    assert dets[0]["confidence"] == 0.9 and dets[0]["class_id"] == 1


def test_to_detections_empty():
    assert to_detections(np.zeros((0, 4)), [], [], NAMES, 10, 10) == []


def test_draw_keeps_size_and_input():
    rgb = np.full((120, 160, 3), 30, np.uint8)
    dets = to_detections([[20, 30, 100, 110]], [0], [0.8], NAMES, 160, 120)
    out = draw(rgb, dets)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert (rgb == 30).all()  # input untouched
    assert (out != 30).any()  # something drawn
    assert (draw(rgb, []) == rgb).all()


def test_load_rgb_variants(tmp_path):
    gray = np.full((8, 6), 7, np.uint8)
    assert load_rgb(gray).shape == (8, 6, 3)
    assert load_rgb(np.ones((4, 4, 3), np.float32)).max() == 255
    assert load_rgb(np.zeros((4, 4, 4), np.uint8)).shape == (4, 4, 3)
    p = tmp_path / "x.png"
    Image.fromarray(np.zeros((5, 9, 3), np.uint8)).save(p)
    assert load_rgb(p).shape == (5, 9, 3)
    assert load_rgb(Image.open(p)).shape == (5, 9, 3)


@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None or not WEIGHTS.exists(),
                    reason="needs ultralytics and exports/yolo11s_ruod/best.pt")
def test_detector_end_to_end():
    from uie.detect import Detector
    det = Detector(WEIGHTS, device="cpu")
    rgb = np.random.default_rng(0).integers(0, 255, (360, 480, 3), dtype=np.uint8)
    boxed, dets = det.detect(rgb)
    assert boxed.shape == rgb.shape
    for d in dets:
        assert set(d) == {"box", "class", "class_id", "confidence"}
        assert d["class"] in det.names.values()
