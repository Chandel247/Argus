# Underwater Image Enhancement — Baselines

Reproducible baseline benchmark for our underwater image enhancement (UIE)
model. The protocol follows the unified benchmark in Cong et al.,
*A Comprehensive Survey on Underwater Image Enhancement Based on Deep Learning*
(arXiv:2405.19684), Sec. IV.

## Datasets

| Dataset | Source (Kaggle) | Use | Split |
|---|---|---|---|
| UIEB | `naomirory2/uieb-final` | train / test (full-reference) | 800 / 90 (T90), seed 42 |
| UIEB C60 | same | test (no-reference) | 60 |
| LSUI | `noureldin199/lsui-large-scale-underwater-image-dataset` | train / test (full-reference) | 3879 / 400 (L400) |
| RUOD | `naomirory2/ruod-final` | external test (no-reference + detection mAP) | 9800 / 4200, + blur/color/light ×100 |

Split files live in `splits/` and are the single source of truth: every method
is trained and evaluated on exactly these filenames.

```bash
bash scripts/download_data.sh            # all datasets into data/
bash scripts/download_data.sh lsui       # just one
```

## Protocol

- Train/test resolution 256×256, batch size 8.
- Augmentation: random horizontal flip, vertical flip, random crop.
- Per-method optimizer / LR schedule from the original paper; no extra tuning.
- No backbones pretrained on external data.
- One shared metric implementation for all methods.

**Training regimes**
- A: train UIEB-800 → test T90, C60, L400, RUOD
- B: train LSUI-3879 → test L400, T90, C60, RUOD

**Metrics**
- Full-reference: PSNR, SSIM, LPIPS
- No-reference: UIQM, UCIQE, URanker, MUSIQ/NIQE
- Downstream: detector trained on raw RUOD-train; mAP on raw vs. enhanced RUOD-test
- Cost: params, FLOPs, runtime @256×256

## Evaluation

All methods are scored with `uie/metrics.py` through one script. Predictions
are matched to references by filename stem; both are resized to 256×256
(`--size 0` for native resolution).

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

# full-reference test set (default metrics: psnr, ssim, lpips, uiqm, uciqe)
.venv/bin/python scripts/evaluate.py --pred results/<method>/uieb_t90 \
    --gt data/uieb/reference-890 --split splits/uieb_test.txt \
    --name <method> --dataset uieb_t90

# no-reference test set, all no-reference metrics
.venv/bin/python scripts/evaluate.py --pred results/<method>/c60 \
    --metrics uiqm,uciqe,musiq,niqe,uranker --name <method> --dataset uieb_c60
```

Writes `results/<method>/<dataset>.csv` (per image) and `.json` (means).
Score the raw inputs too (`--pred data/uieb/raw-890 --name input`) so each table
has an Input row.

Metric notes:
- **UIQM** is a vectorized port of the FUnIE-GAN `uqim_utils.py` implementation
  used by many papers, checked against a literal transcription in `tests/`.
- **UCIQE** follows Yang & Sowmya (2015) with correctly zero-centered CIELab;
  absolute values are lower than toolboxes that keep OpenCV's +128 a/b offset,
  so compare UCIQE only within this benchmark.
- Deep metrics run on CUDA if available, else CPU (pyiqa's NIQE does not
  support Apple MPS).

```bash
.venv/bin/python -m pytest -q tests
```

## Baselines

Third-party code is cloned at a pinned commit into `baselines/<name>/` (not
vendored) and wrapped by an adapter in `uie/baselines/`.

| Baseline | Adapter | Notes |
|---|---|---|
| Semi-UIR (CVPR'23) | `uie/baselines/semi_uir.py` | mmcv DCN → torchvision shim; unlabeled set = 800 LSUI-train inputs; checkpoint selection on `splits/uieb_val.txt` (50 held out of UIEB train), T90 never used for selection |

## Training on Kaggle

Training runs as private Kaggle script kernels (2×T4), driven locally:

```bash
python3 kaggle/launch.py sync-code               # upload git HEAD as private dataset argus-code
python3 kaggle/launch.py push semi-uir-check     # ~15 min sanity run (official weights + smoke train)
python3 kaggle/launch.py push semi-uir-uieb      # protocol run A, session 1
python3 kaggle/launch.py status semi-uir-uieb
python3 kaggle/launch.py next semi-uir-uieb      # after a session ends: fetch, save state, resume
```

Each session stops cleanly after `max_hours` (6 h for `semi-uir-uieb`). `next`
downloads its output to `runs/<job>/<stamp>/`, re-uploads the checkpoint, the
reliable bank and the cached illumination maps as a private
`argus-state-<job>` dataset, and starts the following session from it.

Robustness, after a session was SIGKILLed mid-epoch with no traceback:
- `run/mem.log` records process-tree RSS and the container's cgroup memory and
  OOM-kill counter every minute (also summarised per epoch in `metrics.jsonl`);
- the kernel restarts a SIGKILLed trainer from its last checkpoint (up to
  `kill_retries` times) and logs the cgroup OOM count;
- the bank and illumination maps are single files (`bank.pack`, `la.pack`;
  uncompressed tars), and `fetch` downloads only the files needed to resume,
  export and report, retrying incomplete downloads;
- illumination maps are computed once, in parallel, then restored from `la.pack`.

## Export and demo

```bash
# pick the best checkpoint (highest val PSNR) across fetched sessions and export it
.venv/bin/python scripts/export_best.py --model semi_uir --kaggle-job semi-uir-uieb
# before any training has finished: export the official released weights
.venv/bin/python scripts/export_best.py --model semi_uir --official baselines/semi_uir/pretrained/model.pth

bash scripts/fetch_demo_examples.sh               # a few UIEB T90 images for the demo
.venv/bin/python demo/app.py --model semi_uir     # http://127.0.0.1:7860
```

The exporter only reads `ckpt/best.pt` / `metrics.jsonl` (written atomically
by the trainer) and only fetches *finished* Kaggle sessions (including ones that ended in an error), so it never
interferes with a running one. It replaces the current export only with a
better one, switching `exports/<model>/current.json` atomically; the demo
watches that file and hot-reloads the new weights. `--watch 600` repeats the
export periodically.

## Layout

```
configs/    dataset paths and sources
splits/     train/test filename lists
uie/        shared metrics, baseline adapters, compatibility shims
scripts/    data, splits, training, inference, evaluation, export
kaggle/     Kaggle kernel entry point and launcher
demo/       Flask demo website
tests/      metric, shim, export and demo tests
runs/       fetched Kaggle session outputs (gitignored)
exports/    exported weights served by the demo (gitignored)
baselines/  cloned third-party baseline repos (gitignored)
data/       datasets (gitignored)
```
