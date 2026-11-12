#!/usr/bin/env bash
# Runs every experiment of the paper in order, then builds the tables and figures. Every step skips the outputs that
# already exist in $RESULTS, so the script can be stopped and started again. On macOS the machine is kept awake.
#
# The repository ships the results of the paper in results/: recompute everything in another folder with
#   RESULTS=results_rerun bash experiments/scripts/run_all.sh
# and compare the two. About 14 hours with JOBS=2 on a laptop. Optional variables: see common.sh.
set -euo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"

# Keep a Mac awake for the whole run
if [ "$(uname)" = "Darwin" ] && [ -z "${SHADE_AWAKE:-}" ] && command -v caffeinate > /dev/null; then
  export SHADE_AWAKE=1
  exec caffeinate -is bash "$HERE/run_all.sh" "$@"
fi

bash "$HERE/01_timing.sh"        # first and alone on the machine, for clean time measurements
bash "$HERE/02_detection.sh"
bash "$HERE/03_drift_sweep.sh"
bash "$HERE/04_drift_stream.sh"
bash "$HERE/05_tables_figures.sh"
