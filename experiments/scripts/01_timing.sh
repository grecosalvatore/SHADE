#!/usr/bin/env bash
# Execution time per window of DriftLens and SHADE (K = 64, B = 2) on UC01, for window sizes from 250 to 5000 in
# steps of 250: 5 runs of 20 windows per size, single thread (the execution-time figure).
# Run it ALONE: no other experiment may run on the machine at the same time.
# Output: $RESULTS/raw/timing/UC01.json (about 5 minutes); skipped if it exists.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_inputs

log "timing: UC01, 1 thread (log: $LOG_DIR/timing.log)"
py experiments/run_timing.py --uc 01 --out-dir "$RAW/timing" > "$LOG_DIR/timing.log" 2>&1
log "timing: done ($RAW/timing)"
