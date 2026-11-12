#!/usr/bin/env bash
# Tables (LaTeX and Markdown) and figures of the paper from the results in $RESULTS/raw. It does not need the
# embeddings. Takes less than a minute.
# Output: $RESULTS/tables/ and $RESULTS/figures/.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

py experiments/make_tables.py --raw-dir "$RAW" --out-dir "$RESULTS/tables"
py experiments/make_figures.py --raw-dir "$RAW" --out-dir "$RESULTS/figures"
