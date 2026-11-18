"""RUOD detection splits (seed 42), from the COCO annotation files.

RUOD ships 9800 train / 4200 test images. We add:
  ruod_val.txt          980 images held out of train (10%) to pick the best
                        epoch, so the test set never influences training
  ruod_train_fit.txt    the other 8820 train images
  ruod_test.txt         all 4200 test images
  ruod_bench400.txt     400 test images: the shared detection benchmark
                        (enhance these, then detect; see docs/ruod_benchmark.md)
  ruod_env_{blur,color,light}.txt   RUOD's 100-image degradation subsets
                        (all part of the test set)
The benchmark is a seeded random sample of the test set, accepted only if
every class keeps its test-set share of images within MAX_DEV and has at
least MIN_IMAGES images; it also writes per-class statistics.

python scripts/make_ruod_splits.py --ann data/ruod_ann
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
VAL_FRAC = 0.10
N_BENCH = 400
MAX_DEV = 0.015    # absolute difference in the share of images containing a class
MIN_IMAGES = 20
ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "splits"


def load(path: Path):
    d = json.loads(path.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_id = {i["id"]: i["file_name"] for i in d["images"]}
    classes = {n: Counter() for n in by_id.values()}  # file -> boxes per class
    for a in d["annotations"]:
        classes[by_id[a["image_id"]]][cats[a["category_id"]]] += 1
    return cats, sorted(by_id.values()), classes


def write(name, names):
    (SPLITS / name).write_text("\n".join(names) + "\n")


def shares(names, classes, cat_names):
    n = len(names)
    return {c: sum(c in classes[f] for f in names) / n for c in cat_names}


def pick_bench(test, classes, cat_names):
    full = shares(test, classes, cat_names)
    for attempt in range(1000):  # first sample that passes; attempt 0 almost always does
        rng = random.Random(SEED + attempt)
        bench = sorted(rng.sample(test, N_BENCH))
        s = shares(bench, classes, cat_names)
        counts = {c: round(s[c] * N_BENCH) for c in cat_names}
        if all(abs(s[c] - full[c]) <= MAX_DEV for c in cat_names) and min(counts.values()) >= MIN_IMAGES:
            return bench, SEED + attempt, full, s
    raise SystemExit("no balanced sample found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, required=True, help="dir with RUOD_ANN/Environmet_ANN json files")
    a = ap.parse_args()
    find = lambda name: next(a.ann.rglob(name))

    cats, train, _ = load(find("instances_train.json"))
    _, test, classes = load(find("instances_test.json"))
    cat_names = [cats[k] for k in sorted(cats)]

    val = sorted(random.Random(SEED).sample(train, round(VAL_FRAC * len(train))))
    fit = sorted(set(train) - set(val))
    write("ruod_val.txt", val)
    write("ruod_train_fit.txt", fit)
    write("ruod_test.txt", test)

    env = {}
    for e in ("blur", "color", "light"):
        _, names, _ = load(find(f"instances_{e}.json"))
        assert set(names) <= set(test), f"{e} subset is not inside the test set"
        env[e] = names
        write(f"ruod_env_{e}.txt", names)

    bench, seed, full, s = pick_bench(test, classes, cat_names)
    write(f"ruod_bench{N_BENCH}.txt", bench)
    stats = {
        "seed": seed, "n_images": len(bench),
        "n_boxes": sum(sum(classes[f].values()) for f in bench),
        "n_boxes_test": sum(sum(classes[f].values()) for f in test),
        "classes": {c: {"images": round(s[c] * N_BENCH), "boxes": sum(classes[f][c] for f in bench),
                        "image_share": round(s[c], 4), "image_share_in_test": round(full[c], 4)}
                    for c in cat_names},
        "in_env_subsets": {e: len(set(bench) & set(n)) for e, n in env.items()},
    }
    (SPLITS / f"ruod_bench{N_BENCH}_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"train fit/val: {len(fit)}/{len(val)}; test: {len(test)}; bench: {len(bench)} (seed {seed})")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
