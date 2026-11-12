#!/usr/bin/env bash
# Downloads the source of the DriftLens baseline, at the commit used for the results of the paper, into
# third_party/drift-lens (only the 'driftlens' package and its license). The DriftLens repository cannot be installed
# with pip, so experiments/baselines.py imports it from this folder. Nothing is downloaded if it is already there.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

COMMIT=0b7b943b128f8e23b56e5cd56fa40bc3dd35119e
URL="https://github.com/grecosalvatore/drift-lens/archive/$COMMIT.tar.gz"
DEST=third_party/drift-lens

if [ -f "$DEST/driftlens/driftlens.py" ] && [ "$(cat "$DEST/COMMIT" 2> /dev/null)" = "$COMMIT" ]; then
  exit 0
fi
echo "Downloading DriftLens (commit ${COMMIT:0:7}) into $DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$URL" -o "$tmp/drift-lens.tar.gz"
rm -rf "$DEST"
mkdir -p "$DEST"
tar -xzf "$tmp/drift-lens.tar.gz" -C "$DEST" --strip-components=1 "drift-lens-$COMMIT/driftlens" "drift-lens-$COMMIT/LICENSE.md"
echo "$COMMIT" > "$DEST/COMMIT"
echo "DriftLens is in $DEST"
