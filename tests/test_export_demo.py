"""Exporter + demo tests with fake checkpoints and a stand-in model (no
third-party baseline code is executed)."""
import io
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "demo"))

import export_best  # noqa: E402


def fake_run(root: Path, session: str, val_psnr: float, epoch: int) -> Path:
    run = root / session / "run"
    (run / "ckpt").mkdir(parents=True)
    sd = {"module.conv.weight": torch.full((2, 2), val_psnr)}
    torch.save({"student": sd, "epoch": epoch, "val_psnr": val_psnr}, run / "ckpt" / "best.pt")
    return run


def export(tmp, *extra):
    export_best.main(["--model", "m", "--out", str(tmp / "exports"), *map(str, extra)])
    return json.loads((tmp / "exports" / "m" / "current.json").read_text())


def test_picks_best_and_never_downgrades(tmp_path):
    a = fake_run(tmp_path, "s1", 20.0, 10)
    b = fake_run(tmp_path, "s2", 22.5, 30)
    cur = export(tmp_path, "--run", a, b)
    assert cur["tag"] == "s2-e030" and cur["val_psnr"] == 22.5 and cur["epoch"] == 30

    sd = torch.load(tmp_path / "exports" / "m" / cur["weights"], weights_only=True)
    assert list(sd) == ["conv.weight"]  # DataParallel prefix stripped
    assert torch.all(sd["conv.weight"] == 22.5)

    worse = fake_run(tmp_path, "s3", 21.0, 50)
    assert export(tmp_path, "--run", worse)["tag"] == "s2-e030"
    assert export(tmp_path, "--run", worse, "--force")["tag"] == "s3-e050"


def test_official_does_not_replace_trained(tmp_path):
    official = tmp_path / "official.pth"
    torch.save({"state_dict": {"module.w": torch.zeros(1)}, "epoch": 200}, official)
    assert export(tmp_path, "--official", official)["source"] == "official"
    run = fake_run(tmp_path, "s1", 18.0, 5)
    assert export(tmp_path, "--run", run)["tag"] == "s1-e005"  # trained beats unscored
    assert export(tmp_path, "--official", official)["tag"] == "s1-e005"


def test_partial_checkpoint_is_ignored(tmp_path):
    run = fake_run(tmp_path, "s1", 20.0, 1)
    (run / "ckpt" / "best.pt.tmp").write_bytes(b"partial write")
    assert export(tmp_path, "--run", run)["tag"] == "s1-e001"


# --------------------------------------------------------------------------- demo
@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    import app as demo_app

    class FakeEnhancer:
        def __init__(self, weights, device="cpu"):
            self.weights = Path(weights)

        def __call__(self, img, size=None, max_side=None):
            h, w = img.shape[:2]
            s = min(1.0, max_side / max(h, w))
            th, tw = int(h * s) // 8 * 8, int(w * s) // 8 * 8
            return 255 - np.asarray(Image.fromarray(img).resize((tw, th)))

    fake = types.SimpleNamespace(Enhancer=FakeEnhancer)
    monkeypatch.setattr(demo_app.baselines, "get", lambda name: fake)
    exports = tmp_path / "exports"
    run = fake_run(tmp_path, "s1", 20.0, 3)
    export_best.main(["--model", "m", "--out", str(exports), "--run", str(run)])
    slot = demo_app.ModelSlot("m", exports, "cpu")
    return demo_app.create_app(slot, max_side=128).test_client(), tmp_path, exports


def png_bytes(w=300, h=200):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_demo_enhance(demo_client):
    client, _, _ = demo_client
    m = client.get("/api/model").get_json()
    assert m["ok"] and m["tag"] == "s1-e003"
    r = client.post("/api/enhance", data={"image": (io.BytesIO(png_bytes()), "x.png")},
                    content_type="multipart/form-data").get_json()
    assert r["ok"] and r["orig_size"] == [300, 200]
    assert max(r["size"]) <= 128 and r["size"][0] % 8 == 0
    assert r["output"].startswith("data:image/jpeg;base64,")


def test_demo_rejects_garbage(demo_client):
    client, _, _ = demo_client
    r = client.post("/api/enhance", data={"image": (io.BytesIO(b"not an image"), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_demo_hot_reloads_new_export(demo_client):
    client, tmp, exports = demo_client
    assert client.get("/api/model").get_json()["tag"] == "s1-e003"
    better = fake_run(tmp, "s2", 23.0, 40)
    export_best.main(["--model", "m", "--out", str(exports), "--run", str(better)])
    pointer = exports / "m" / "current.json"
    os.utime(pointer, (pointer.stat().st_atime, pointer.stat().st_mtime + 5))  # coarse mtime clocks
    assert client.get("/api/model").get_json()["tag"] == "s2-e040"
