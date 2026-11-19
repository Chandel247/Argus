"""Model size and speed of the GAN generator, Semi-UIR and YOLO11s on one GPU.

For each model: parameter count, weight file size, and images/second at
batch 1 (warm-up first, then 100 real images, timed one by one with CUDA
synchronisation). Two speeds per model:
  model     the network's forward pass only, input already a GPU tensor
  pipeline  decoded uint8 image in RAM -> uint8 result, including the model's
            own pre/post-processing (Semi-UIR: illumination map on CPU;
            YOLO: letterbox + NMS; GAN: resize + normalise)
Enhancers run at the benchmark protocol size (256x256), YOLO at 640 (its
training/evaluation size). FP32 everywhere, one GPU.

python scripts/bench_cost.py --gan <argus-gan>/runs/gan/best.pt --semi <ckpt>/best.pt \
    --yolo yolo11s_ruod_best.pt --enh-images <UIEB>/raw-890 --enh-split splits/uieb_test.txt \
    --det-images <RUOD>/RUOD_pic/test --det-split splits/ruod_bench400.txt --out results/cost.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --- GAN generator: architecture as defined in Raghav's notebooks
# (raghavb441/argus-gan, raghavb441/argus-eval), copied unchanged
def _down(cin, cout, bn=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1)]
    if bn:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = _down(3, 32, bn=False); self.d2 = _down(32, 128); self.d3 = _down(128, 256)
        self.d4 = _down(256, 256); self.d5 = _down(256, 256)
        self.u1 = _up(256, 256); self.u2 = _up(512, 256); self.u3 = _up(512, 128); self.u4 = _up(256, 32)
        self.out = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3); d5 = self.d5(d4)
        u = torch.cat([self.u1(d5), d4], 1); u = torch.cat([self.u2(u), d3], 1)
        u = torch.cat([self.u3(u), d2], 1); u = torch.cat([self.u4(u), d1], 1)
        return self.out(u)


def mb(nbytes: float) -> float:
    return round(nbytes / 2 ** 20, 2)


def state_dict_mb(sd: dict) -> float:
    """Size of the weights alone, saved as a plain state_dict."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(sd, f.name)
        size = os.path.getsize(f.name)
    os.unlink(f.name)
    return mb(size)


def timed(fn, inputs, warmup: int) -> dict:
    """Per-image wall time, batch 1, with CUDA synchronisation around each call."""
    for x in inputs[:warmup]:
        fn(x)
    torch.cuda.synchronize()
    times = []
    for x in inputs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = np.array(times)
    return {"images": len(t), "img_per_s": round(len(t) / t.sum(), 1), "ms_mean": round(1000 * t.mean(), 2),
            "ms_median": round(1000 * float(np.median(t)), 2)}


def read_images(folder: Path, split: Path, n: int) -> list[np.ndarray]:
    names = [l.strip() for l in split.read_text().splitlines() if l.strip()][:n]
    return [np.asarray(Image.open(folder / name).convert("RGB")) for name in names]


def bench_gan(path, imgs, size, warmup, dev):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # teammate checkpoint with non-tensor metadata
        ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["G"] if "G" in ck else ck
    G = Generator()
    G.load_state_dict(sd)
    G = G.to(dev).eval()
    small = [np.asarray(Image.fromarray(a).resize((size, size), Image.BICUBIC)) for a in imgs]
    to_t = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(dev)
    tensors = [to_t(a) for a in small]

    @torch.no_grad()
    def pipeline(a):  # same steps as Raghav's enhance_256
        x = to_t(np.asarray(Image.fromarray(a).resize((size, size), Image.BICUBIC)))
        y = G(x)[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
        return ((y + 1) * 127.5).round().astype(np.uint8)

    with torch.no_grad():
        model = timed(G, tensors, warmup)
    return {"params": sum(p.numel() for p in G.parameters()), "checkpoint_file_mb": mb(os.path.getsize(path)),
            "weights_mb": state_dict_mb(G.state_dict()), "input": f"{size}x{size}",
            "model": model, "pipeline": timed(pipeline, imgs, warmup),
            "checkpoint_keys": sorted(k for k in ck if isinstance(k, str))[:12]}


def bench_semi(path, imgs, size, warmup, dev):
    from uie.baselines import semi_uir
    enh = semi_uir.Enhancer(path, device=str(dev))
    m = enh.model
    pairs = []
    for a in imgs:
        x = np.asarray(Image.fromarray(a).resize((size, size), Image.LANCZOS))
        l = np.asarray(Image.fromarray(semi_uir.luminance_estimation(a)).resize((size, size), Image.LANCZOS))
        t = lambda v: torch.from_numpy(v).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        pairs.append((t(x), t(l)))
    with torch.no_grad():
        model = timed(lambda p: m(*p), pairs, warmup)
    return {"params": sum(p.numel() for p in m.parameters()), "checkpoint_file_mb": mb(os.path.getsize(path)),
            "weights_mb": state_dict_mb(m.state_dict()), "input": f"{size}x{size}",
            "model": model, "pipeline": timed(lambda a: enh(a, size=size), imgs, warmup)}


def bench_yolo(path, imgs, imgsz, warmup, dev):
    from ultralytics import YOLO
    y = YOLO(str(path))
    net = y.model
    bgr = [np.ascontiguousarray(a[..., ::-1]) for a in imgs]
    pred = lambda a: y.predict(a, imgsz=imgsz, device=dev.index if dev.type == "cuda" else "cpu",
                               half=False, verbose=False)
    pipeline = timed(pred, bgr, warmup)
    # forward pass alone, on the letterboxed tensors the predictor actually feeds the network
    inner = y.predictor.model  # AutoBackend wrapping the fp32 network
    tensors = []
    for a in bgr:
        im = y.predictor.preprocess([a])
        tensors.append(im)
    with torch.no_grad():
        model = timed(lambda t: inner(t), tensors, warmup)
    return {"params": sum(p.numel() for p in net.parameters()), "checkpoint_file_mb": mb(os.path.getsize(path)),
            "weights_mb": state_dict_mb({k: v.float() for k, v in net.state_dict().items()}),
            "input": f"{imgsz} (letterboxed, long side)", "model": model, "pipeline": pipeline,
            "ultralytics_speed_ms": {k: round(v, 2) for k, v in pred(bgr[0])[0].speed.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gan", required=True)
    ap.add_argument("--semi", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--enh-images", required=True, type=Path)
    ap.add_argument("--enh-split", required=True, type=Path)
    ap.add_argument("--det-images", required=True, type=Path)
    ap.add_argument("--det-split", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    dev = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    enh = read_images(a.enh_images, a.enh_split, a.n)
    det = read_images(a.det_images, a.det_split, a.n)
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "batch": 1, "warmup": a.warmup, "timed_images": a.n, "precision": "fp32",
           "enh_images": f"{a.enh_split.name} (first {len(enh)})", "det_images": f"{a.det_split.name} (first {len(det)})"}
    for name, fn, args in [("gan_generator", bench_gan, (a.gan, enh, a.size)),
                           ("semi_uir", bench_semi, (a.semi, enh, a.size)),
                           ("yolo11s", bench_yolo, (a.yolo, det, a.imgsz))]:
        print(f"== {name}", flush=True)
        res[name] = fn(*args, a.warmup, dev)
        torch.cuda.empty_cache()
        print(json.dumps(res[name], indent=1), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nGPU: {res['gpu']}")
    print(f"{'model':16} {'params':>12} {'file MB':>8} {'weights MB':>10} {'model img/s':>12} {'pipeline img/s':>15}")
    for k in ("gan_generator", "semi_uir", "yolo11s"):
        r = res[k]
        print(f"{k:16} {r['params']:>12,} {r['checkpoint_file_mb']:>8} {r['weights_mb']:>10} "
              f"{r['model']['img_per_s']:>12} {r['pipeline']['img_per_s']:>15}")


if __name__ == "__main__":
    main()
