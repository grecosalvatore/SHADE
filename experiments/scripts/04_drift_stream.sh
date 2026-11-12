#!/usr/bin/env bash
# Drift patterns over a simulated stream (the qualitative figures): UC01 and UC06, 100 windows of 1000 samples,
# sudden / incremental / periodic drift up to 20 %, global and per-label scores of DriftLens and SHADE (K = 64, B = 2),
# seeds of run 0, global and per-label thresholds from 10000 threshold windows. The use cases already computed are
# skipped.
# Output: $RESULTS/raw/drift_stream/UC<id>_mw1000.json (about 10 minutes).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_inputs

log "drift stream: UC01, UC06 (log: $LOG_DIR/drift_stream.log)"
py experiments/run_drift_stream.py --uc 01 06 --window-size 1000 --n-windows 100 --max-drift 0.20 --run 0 \
  --n-th 10000 --out-dir "$RAW/drift_stream" > "$LOG_DIR/drift_stream.log" 2>&1
log "drift stream: done ($RAW/drift_stream)"
