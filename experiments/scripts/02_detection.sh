#!/usr/bin/env bash
# Drift detection (the main table and the variants table): UC01-UC10, window sizes 250, 500, 1000, 2000 (a size is
# skipped when a window does not fit in the smallest label pool: UC02 and UC03 at 2000), 5 runs, 10000 threshold
# windows, 1000 windows per drift percentage (0, 5, 10 %), DriftLens and SHADE with K in {64, 128} and B in {2, 8}.
# Single thread, JOBS processes in parallel. The (use case, window size) pairs already computed are skipped, so the
# script can be stopped and started again.
# Output: $RESULTS/raw/detection/UC<id>_mw<m_w>.json (about 11-12 hours with JOBS=2 on a laptop).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_inputs

# Ordered so that, dealt round-robin to 2 processes, the two loads are similar
log "detection: UC01-UC10, $JOBS processes"
in_parallel detection experiments/run_detection.py "01 03 04 05 07 06 10 08 02 09" --out-dir "$RAW/detection"
log "detection: done ($RAW/detection)"
