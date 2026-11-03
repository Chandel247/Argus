"""Generate split files deterministically (seed 42).

LSUI uses the team's existing 3879/400 split (splits/lsui_*.txt), and RUOD uses
its official train/test folders, so only UIEB is generated here.

Usage:
    python scripts/make_splits.py                              # derived splits only
    python scripts/make_splits.py --uieb-dir data/uieb/raw-890 # also regenerate T90
"""
import argparse
import random
from pathlib import Path

SEED = 42
N_TEST = 90
SPLITS = Path(__file__).resolve().parent.parent / "splits"


N_VAL = 50          # held out of the 800 train images for checkpoint selection
N_UNLABELED = 800   # semi-supervised methods: LSUI-train inputs, GT discarded


def read(name):
    return [l.strip() for l in (SPLITS / name).read_text().splitlines() if l.strip()]


def write(name, names):
    (SPLITS / name).write_text("\n".join(names) + "\n")


def derived_splits():
    """Splits derived from the base split files (no dataset access needed)."""
    train = read("uieb_train.txt")
    val = sorted(random.Random(SEED).sample(train, N_VAL))
    write("uieb_val.txt", val)
    write("uieb_train_fit.txt", sorted(set(train) - set(val)))
    unl = sorted(random.Random(SEED).sample(read("lsui_train.txt"), N_UNLABELED))
    write("unlabeled_lsui800.txt", unl)
    print(f"UIEB fit/val: {len(train) - N_VAL}/{N_VAL}; unlabeled (LSUI train): {N_UNLABELED}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uieb-dir", help="folder with the 890 raw UIEB images (regenerates T90)")
    args = ap.parse_args()

    if args.uieb_dir:
        names = sorted(p.name for p in Path(args.uieb_dir).iterdir() if p.suffix.lower() == ".png")
        assert len(names) == 890, f"expected 890 UIEB images, found {len(names)}"
        rng = random.Random(SEED)
        test = sorted(rng.sample(names, N_TEST))
        train = sorted(set(names) - set(test))
        SPLITS.mkdir(exist_ok=True)
        write("uieb_train.txt", train)
        write("uieb_test.txt", test)
        print(f"UIEB: {len(train)} train / {len(test)} test (seed={SEED})")
    derived_splits()


if __name__ == "__main__":
    main()
