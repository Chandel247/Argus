#!/usr/bin/env bash
# Download all datasets into data/. Requires the Kaggle CLI with credentials
# (~/.kaggle/credentials.json). On Kaggle Notebooks, attach the same datasets
# as inputs instead and point data_root at /kaggle/input.
#
# Usage: bash scripts/download_data.sh [uieb|lsui|ruod ...]   (default: all)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"

src() {
  case "$1" in
    uieb) echo "naomirory2/uieb-final" ;;
    lsui) echo "noureldin199/lsui-large-scale-underwater-image-dataset" ;;
    ruod) echo "naomirory2/ruod-final" ;;
    *) echo "unknown dataset: $1" >&2; exit 1 ;;
  esac
}

[ $# -eq 0 ] && set -- uieb lsui ruod

for name in "$@"; do
  ref="$(src "$name")"
  echo ">> $name  ($ref)"
  mkdir -p "$DATA/$name"
  kaggle datasets download "$ref" -p "$DATA/$name" --unzip
done
