# Our enhancers: GAN and L1-only (Raghav)

A FUnIE-GAN-style conditional GAN (Islam et al., IEEE RA-L 2020) and an L1-only
ablation that uses the same generator. The adapter used by `scripts/infer.py`
and the demo is `uie/baselines/funie_gan.py` (registry names `gan` and `l1`).

## Model

- **Generator:** U-Net, 5 stride-2 down stages (4×4 conv, BatchNorm except the
  first, LeakyReLU 0.2) and 5 up stages (4×4 transposed conv, BatchNorm, ReLU)
  with skip connections; tanh output. 7.02 M parameters.
- **Discriminator (GAN run only):** PatchGAN on the raw image concatenated with
  the enhanced or reference image (6 channels); a 16×16 grid of scores for a
  256×256 input, each with a 78×78 receptive field. 0.69 M parameters.
- **Losses:** GAN run = LSGAN + 0.7·L1 + 0.3·VGG-19 content (conv5_2 features,
  MSE). L1-only run = L1 alone. Weights 0.7 / 0.3 from the FUnIE-GAN paper.

## Training (`enhance/train.py`)

| Setting | Value |
|---|---|
| Data | `splits/uieb_train_fit.txt` (750 pairs); best checkpoint by PSNR on `splits/uieb_val.txt` (50) |
| Input | 256×256 (bicubic), pixels scaled to [-1, 1] |
| Augmentation | random horizontal flip only (same flip on both images of a pair) |
| Optimiser | Adam, lr 3e-4, betas (0.5, 0.999), for G and D |
| Schedule | batch 8, 200 epochs, seed 42, one Tesla T4 (Kaggle) per run |
| Checkpoints | `best.pt` (highest val PSNR), `last.pt` (resume), written atomically |

```bash
python enhance/train.py --mode gan --epochs 200 --out runs/gan
python enhance/train.py --mode l1  --epochs 200 --out runs/l1
```

The script was run in a Kaggle notebook: the dataset path (`BASE`) points at the
attached `naomirory2/uieb-final` dataset and the split files are located under
`/kaggle/input`. Change those two lines to run elsewhere.

Note: the shared protocol in the main README also lists vertical flips and
random crops; these two runs used horizontal flips only.

| Run | Best epoch | Best val PSNR | Time / epoch | Total |
|---|---:|---:|---:|---:|
| GAN | 195 | 20.17 dB | ~40 s | ~2.2 h |
| L1-only | 170 | 20.56 dB | ~8 s | ~27 min |

## Results (scored with `scripts/evaluate.py`, 256×256, same protocol as Semi-UIR)

| Test set | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | UIQM ↑ | UCIQE ↑ |
|---|---|---:|---:|---:|---:|---:|
| UIEB T90 | Input | 16.94 | 0.761 | 0.252 | 2.579 | 0.248 |
| UIEB T90 | GAN | 19.78 | 0.839 | 0.162 | 3.188 | 0.296 |
| UIEB T90 | L1-only | 20.20 | 0.864 | 0.161 | 3.194 | 0.292 |
| LSUI L400 | Input | 18.24 | 0.785 | 0.347 | 2.526 | 0.272 |
| LSUI L400 | GAN | 19.07 | 0.778 | 0.263 | 3.140 | 0.293 |
| LSUI L400 | L1-only | 19.11 | 0.799 | 0.258 | 3.144 | 0.291 |
| UIEB C60 | Input | – | – | – | 1.959 | 0.233 |
| UIEB C60 | GAN | – | – | – | 2.954 | 0.269 |
| UIEB C60 | L1-only | – | – | – | 2.972 | 0.267 |

For RUOD detection, the 400 bench images (`splits/ruod_bench400.txt`) were
enhanced at native resolution (`size=None`: pad to a multiple of 32, crop back);
detector scores are in `results/det_bench400/`.

## Weights and the demo

The checkpoints (`gan_best.pt`, `l1_best.pt`, ~27 MB each) are not in the repo.
They come from the `argus-gan` Kaggle notebook output (`runs/gan/best.pt`,
`runs/l1/best.pt`). To serve one in the demo:

```bash
mkdir -p exports/gan/e195 && cp gan_best.pt exports/gan/e195/weights.pt
echo '{"weights": "e195/weights.pt", "tag": "gan-e195"}' > exports/gan/current.json
python demo/app.py --model gan            # or the same steps with "l1"

python scripts/infer.py --model gan --weights gan_best.pt \
    --input data/uieb/raw-890 --split splits/uieb_test.txt --out results/gan/uieb_t90
```
