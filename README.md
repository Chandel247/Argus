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

## Layout

```
configs/    dataset paths and sources
splits/     train/test filename lists
scripts/    data download, split generation
baselines/  cloned third-party baseline repos (gitignored)
data/       datasets (gitignored)
```
