"""Generate the UIEB 800/90 split (T90) deterministically.

LSUI uses the team's existing 3879/400 split (splits/lsui_*.txt), and RUOD uses
its official train/test folders, so only UIEB is generated here.

Usage:
    python scripts/make_splits.py --uieb-dir data/uieb/raw-890
"""
import argparse
import random
from pathlib import Path

SEED = 42
N_TEST = 90
SPLITS = Path(__file__).resolve().parent.parent / "splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uieb-dir", required=True, help="folder with the 890 raw UIEB images")
    args = ap.parse_args()

    names = sorted(p.name for p in Path(args.uieb_dir).iterdir() if p.suffix.lower() == ".png")
    assert len(names) == 890, f"expected 890 UIEB images, found {len(names)}"

    rng = random.Random(SEED)
    test = sorted(rng.sample(names, N_TEST))
    train = sorted(set(names) - set(test))

    SPLITS.mkdir(exist_ok=True)
    (SPLITS / "uieb_train.txt").write_text("\n".join(train) + "\n")
    (SPLITS / "uieb_test.txt").write_text("\n".join(test) + "\n")
    print(f"UIEB: {len(train)} train / {len(test)} test (seed={SEED})")


if __name__ == "__main__":
    main()
