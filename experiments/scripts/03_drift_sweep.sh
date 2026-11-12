#!/usr/bin/env bash
# Drift score vs drift percentage (the drift-severity table): UC01-UC10, window size 1000, drift from 0 to 100 % in
# steps of 5 %, 200 windows per percentage, 5 runs, DriftLens (global score) and SHADE (K = 64, B = 2); thresholds
# from 10000 threshold windows per run. Single thread, JOBS processes in parallel. The use cases already computed
# are skipped.
# Output: $RESULTS/raw/drift_sweep/UC<id>_mw1000.json (about 2 hours with JOBS=2 on a laptop).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_inputs

log "drift sweep: UC01-UC10, window size 1000, $JOBS processes"
in_parallel drift_sweep experiments/run_drift_sweep.py "01 06 02 07 03 08 04 09 05 10" --window-size 1000 \
  --out-dir "$RAW/drift_sweep"
log "drift sweep: done ($RAW/drift_sweep)"
