"""Build the YOLO version of RUOD from the original images + COCO annotations.

  train     splits/ruod_train_fit.txt  (8820)  resized copies, longer side <= --max-side
  val       splits/ruod_val.txt        (980)   resized copies (picks the best epoch)
  test      splits/ruod_test.txt       (4200)  originals (symlinks)
  bench400  splits/ruod_bench400.txt   (400)   originals: the shared benchmark
  blur, color, light  splits/ruod_env_*.txt  (100 each) originals
Training copies are downscaled once because ~1200 training images are 4K and
decoding them every epoch would starve the GPUs on Kaggle's 4 CPUs; training
runs at 640 px, so a 1024 px copy loses nothing. Evaluation sets keep the
original files, exactly what enhancement methods receive.

Writes data.yaml (train/val) and data_<set>.yaml for each evaluation set, plus
prepare.json with counts (links, copies, size mismatches).

python scripts/prepare_ruod_yolo.py --ruod data/ruod --out /tmp/ruod_yolo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uie import ruod  # noqa: E402

SPLITS = ROOT / "splits"
SETS = {  # name: (split file, image folder, annotation file, resize for training)
    "train": ("ruod_train_fit.txt", "RUOD_pic/train", "RUOD_ANN/instances_train.json", True),
    "val": ("ruod_val.txt", "RUOD_pic/train", "RUOD_ANN/instances_train.json", True),
    "test": ("ruod_test.txt", "RUOD_pic/test", "RUOD_ANN/instances_test.json", False),
    "bench400": ("ruod_bench400.txt", "RUOD_pic/test", "RUOD_ANN/instances_test.json", False),
    "blur": ("ruod_env_blur.txt", "RUOD_pic/test", "RUOD_ANN/instances_test.json", False),
    "color": ("ruod_env_color.txt", "RUOD_pic/test", "RUOD_ANN/instances_test.json", False),
    "light": ("ruod_env_light.txt", "RUOD_pic/test", "RUOD_ANN/instances_test.json", False),
}
EVAL_SETS = ("test", "bench400", "blur", "color", "light")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruod", type=Path, required=True, help="dir with RUOD_pic/ and RUOD_ANN/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--limit", type=int, help="first N images per set (smoke tests)")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    coco, report = {}, {}
    for name in a.sets.split(","):
        split, folder, ann, resize = SETS[name]
        if ann not in coco:
            coco[ann] = ruod.load_coco(a.ruod / ann)
        names, images = coco[ann]
        files = ruod.read_split(SPLITS / split)[: a.limit]
        t0 = time.time()
        st = ruod.build_set(a.out, name, a.ruod / folder, files, images,
                            max_side=a.max_side if resize else None, workers=a.workers)
        st["sec"] = round(time.time() - t0, 1)
        report[name] = st
        print(f"{name}: {st}", flush=True)
        if name in EVAL_SETS:
            ruod.write_yaml(a.out, names, {"train": f"images/{name}", "val": f"images/{name}"},
                            filename=f"data_{name}.yaml")
    ruod.write_yaml(a.out, names, {"train": "images/train", "val": "images/val", "test": "images/test"})
    (a.out / "prepare.json").write_text(json.dumps(report, indent=2) + "\n")
    bad = sum(st.get("size_mismatch", 0) for st in report.values())
    if bad:
        print(f"WARNING: {bad} images differ in size from their annotations; their boxes may be wrong")


if __name__ == "__main__":
    main()
