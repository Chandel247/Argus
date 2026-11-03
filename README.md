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

## Layout

```
configs/    dataset paths and sources
splits/     train/test filename lists
uie/        shared metric implementations
scripts/    data download, split generation, evaluation
tests/      metric tests
baselines/  cloned third-party baseline repos (gitignored)
data/       datasets (gitignored)
```
