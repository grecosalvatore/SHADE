#!/usr/bin/env bash
# Downloads the embeddings of the 10 use cases from Google Drive (one archive, about 2.7 GB), unzips them into
# data/embeddings and verifies their checksums.
#
# The SHA-256 of the archive is checked before unzipping, and the checksums of the files after.
#
# usage: bash data/download.sh                download with gdown (pip install gdown), verify, unzip, verify
#        bash data/download.sh ARCHIVE.zip    the same for an archive downloaded by hand
#                                             (https://drive.google.com/file/d/1mP2wBbt5I4ryFdWgkolC9msms_xqEJWv)
set -euo pipefail

FILE_ID="1mP2wBbt5I4ryFdWgkolC9msms_xqEJWv"  # Google Drive id of shade_embeddings.zip
ARCHIVE_SHA256="202a4a0a8db36d3f7e3ef60e2b637d3590f656f8ea0d66f8a83739cabbc16306"
ARCHIVE=shade_embeddings.zip

# An archive given by the user is located before moving to the data folder
given=""
if [ $# -ge 1 ]; then
  if [ ! -f "$1" ]; then
    echo "Archive not found: $1" >&2
    exit 1
  fi
  given="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
fi
cd "$(dirname "${BASH_SOURCE[0]}")"

sha256() {
  if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# Checks the archive before unzipping it
check_archive() {
  local digest
  digest="$(sha256 "$1")"
  if [ "$digest" != "$ARCHIVE_SHA256" ]; then
    echo "$1: SHA-256 $digest does not match the expected $ARCHIVE_SHA256 (incomplete or corrupted download)" >&2
    exit 1
  fi
}

verify() {
  if command -v sha256sum > /dev/null; then
    (cd embeddings && sha256sum --quiet -c SHA256SUMS)
  else
    (cd embeddings && shasum -a 256 --quiet -c SHA256SUMS)
  fi
}

if [ -f embeddings/SHA256SUMS ] && verify > /dev/null 2>&1; then
  echo "data/embeddings is already complete"
  exit 0
fi

if [ -n "$given" ]; then
  check_archive "$given"
  unzip -q -o "$given" -d .
else
  if ! command -v gdown > /dev/null; then
    echo "gdown not found: install it with 'pip install gdown' (it is part of the [experiments] extra)" >&2
    exit 1
  fi
  gdown "$FILE_ID" -O "$ARCHIVE"
  check_archive "$ARCHIVE"
  unzip -q -o "$ARCHIVE" -d .
  rm "$ARCHIVE"
fi

verify
echo "Embeddings ready in data/embeddings ($(find embeddings -name '*.hdf5' | wc -l | tr -d ' ') files, checksums verified)"
