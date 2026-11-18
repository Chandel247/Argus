# RUOD detection benchmark (400 test images)

The shared set for testing how underwater image enhancement affects object
detection. Enhance these 400 images with your method, run the fine-tuned YOLO
detector on them, and compare its mAP with the same detector on the original
images.

**The file list:** [`splits/ruod_bench400.txt`](../splits/ruod_bench400.txt)
(400 filenames from `RUOD_pic/test/`, one per line). Per-class statistics:
[`splits/ruod_bench400_stats.json`](../splits/ruod_bench400_stats.json).
Regenerate both with `python scripts/make_ruod_splits.py --ann <RUOD annotations dir>`
(seed-controlled, so the result is identical).

## How it was chosen

- Drawn only from RUOD's **test** set (4,200 images), never from train, so the
  detector has not seen any of them.
- A seeded random sample (`random.Random(127).sample`), accepted only if every
  class keeps its test-set share of images within 1.5 percentage points and
  has at least 20 images. Seeds 42 to 126 were tried first and rejected by that
  rule; 127 is the first that passes.
- 400 images, 2,142 boxes (9.5% of test images, 9.3% of test boxes).

| Class | Images | Boxes | Share of images (bench) | Share of images (full test) |
| --- | ---: | ---: | ---: | ---: |
| holothurian | 74 | 236 | 18.5% | 18.0% |
| echinus | 74 | 339 | 18.5% | 19.2% |
| scallop | 43 | 202 | 10.7% | 10.4% |
| starfish | 74 | 234 | 18.5% | 18.6% |
| fish | 88 | 377 | 22.0% | 23.0% |
| corals | 74 | 229 | 18.5% | 19.0% |
| diver | 94 | 179 | 23.5% | 22.2% |
| cuttlefish | 82 | 147 | 20.5% | 19.3% |
| turtle | 92 | 141 | 23.0% | 21.8% |
| jellyfish | 31 | 58 | 7.7% | 7.7% |

Images often contain several classes, so the image counts add up to more than 400.
32 of the 400 also belong to RUOD's degradation subsets (12 blur, 8 colour,
12 light), listed in `splits/ruod_env_{blur,color,light}.txt`.

## Other RUOD splits

| File | Images | Use |
| --- | ---: | --- |
| `splits/ruod_train_fit.txt` | 8,820 | Detector training |
| `splits/ruod_val.txt` | 980 | Held out of RUOD train to pick the best epoch |
| `splits/ruod_test.txt` | 4,200 | Full RUOD test set |
| `splits/ruod_bench400.txt` | 400 | This benchmark (subset of test) |
| `splits/ruod_env_{blur,color,light}.txt` | 100 each | RUOD degradation subsets (subsets of test) |

## How to score your method

1. Enhance the 400 images and save each one under the **same file stem**
   (`000123.jpg` can become `000123.png`).
2. Keep the whole frame: resizing is fine, cropping or padding is not, because
   the boxes are stored relative to the full image. Enhancing at native
   resolution is best; a 256x256 output is scored correctly, but the detector
   loses detail and mAP drops for that reason alone.
3. Run the detector on your folder and on the originals:

```bash
pip install ultralytics==8.4.180
python scripts/eval_detection.py --weights <yolo best.pt> \
    --images <your enhanced folder> --ann <RUOD>/RUOD_ANN/instances_test.json \
    --split splits/ruod_bench400.txt --name <your_method> --dataset ruod_bench400
python scripts/eval_detection.py --weights <yolo best.pt> \
    --images <RUOD>/RUOD_pic/test --ann <RUOD>/RUOD_ANN/instances_test.json \
    --split splits/ruod_bench400.txt --name raw --dataset ruod_bench400
```

Each run writes `results/<name>/ruod_bench400_det.json` with mAP50, mAP50-95,
precision, recall and per-class AP. Use the same `--imgsz` (default 640) for
every method.

The detector weights (`best.pt`) come from the `yolo-ruod` Kaggle job; ask for
the current file or fetch it with `python kaggle/launch.py fetch yolo-ruod`.

## Reference detector (raw images)

YOLO11s (COCO-pretrained), fine-tuned 100 epochs on `ruod_train_fit.txt`
at 640 px, batch 32, 2×T4 (2.3 h). `best.pt` is the epoch with the best
fitness on `ruod_val.txt` (epoch 79: val mAP50 0.874, mAP50-95 0.649);
SHA-256 starts `a82f4674a95f30f0`. Scores on the **original** images, i.e. the
"raw" row every enhancement method is compared against:

| Set | Images | mAP50 | mAP50-95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| bench400 | 400 | 0.884 | 0.654 | 0.887 | 0.824 |
| test (full) | 4,200 | 0.864 | 0.636 | 0.869 | 0.802 |
| blur | 100 | 0.848 | 0.638 | 0.837 | 0.766 |
| color | 100 | 0.833 | 0.626 | 0.902 | 0.787 |
| light | 100 | 0.718 | 0.529 | 0.707 | 0.742 |

Per-class AP50 / AP50-95:

| Class | bench400 | test | blur | color | light |
| --- | ---: | ---: | ---: | ---: | ---: |
| holothurian | 0.813 / 0.505 | 0.823 / 0.518 | 0.883 / 0.510 | 0.821 / 0.651 | 0.247 / 0.173 |
| echinus | 0.899 / 0.550 | 0.908 / 0.540 | 0.944 / 0.556 | 0.923 / 0.616 | 0.884 / 0.487 |
| scallop | 0.801 / 0.513 | 0.829 / 0.530 | 0.782 / 0.519 | 0.586 / 0.344 | 0.567 / 0.224 |
| starfish | 0.895 / 0.547 | 0.902 / 0.572 | 0.894 / 0.578 | 0.922 / 0.667 | 0.947 / 0.630 |
| fish | 0.770 / 0.526 | 0.773 / 0.549 | 0.848 / 0.630 | 0.520 / 0.341 | 0.420 / 0.259 |
| corals | 0.822 / 0.628 | 0.733 / 0.556 | 0.274 / 0.228 | 0.751 / 0.498 | 0.373 / 0.267 |
| diver | 0.955 / 0.767 | 0.942 / 0.753 | 0.944 / 0.690 | 0.954 / 0.751 | 0.922 / 0.696 |
| cuttlefish | 0.987 / 0.891 | 0.975 / 0.856 | 0.995 / 0.967 | 0.973 / 0.876 | 0.905 / 0.876 |
| turtle | 0.956 / 0.817 | 0.972 / 0.858 | 0.919 / 0.812 | 0.995 / 0.874 | 0.994 / 0.934 |
| jellyfish | 0.941 / 0.793 | 0.787 / 0.625 | 0.995 / 0.895 | 0.887 / 0.637 | 0.919 / 0.739 |

bench400 tracks the full test set closely (mAP50 +0.020, mAP50-95 +0.018),
so it is a fair stand-in. The blur/color/light subsets have only 100 images
each, so a class with a handful of boxes there swings a lot; compare their
overall mAP, not single classes. Poor lighting is where the detector
struggles most (mAP50 0.718), which is where enhancement has the most room to
help.
