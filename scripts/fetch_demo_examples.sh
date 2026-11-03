#!/usr/bin/env bash
# Download a few UIEB T90 test inputs (never used in training or checkpoint
# selection) into demo/static/examples for the demo's example strip.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/demo/static/examples"
N="${1:-8}"
mkdir -p "$DEST"
head -n "$N" "$ROOT/splits/uieb_test.txt" | while read -r name; do
  [ -f "$DEST/$name" ] && continue
  kaggle datasets download naomirory2/uieb-final -f "raw-890/$name" -p "$DEST" -q >/dev/null
done
ls "$DEST"
