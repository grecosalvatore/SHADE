# Shared setup, sourced by the scripts of this folder (not meant to be run directly).
#
# Optional environment variables:
#   PYTHON   Python interpreter (default: python)
#   JOBS     parallel processes of the detection and drift-sweep experiments (default: 2)
#   NICE     niceness of every process (default: 20, the lowest priority)
#   RESULTS  results folder (default: results). The experiments skip the outputs that already exist there, and the
#            repository ships the results of the paper: set another folder, e.g. RESULTS=results_rerun, to recompute.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # repository root: every path below is relative to it

PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-2}"
NICE="${NICE:-20}"
RESULTS="${RESULTS:-results}"
RAW="$RESULTS/raw"
LOG_DIR="$RESULTS/logs"
mkdir -p "$LOG_DIR"

# Single-threaded linear algebra: reproducible times, and several processes can run in parallel without competing
# for the cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

py() { nice -n "$NICE" "$PYTHON" -u "$@"; }
log() { echo "$(date '+%Y-%m-%d %H:%M') $*"; }

# Inputs of the experiments: the embeddings and the DriftLens baseline
check_inputs() {
  if [ ! -d data/embeddings/UC01 ]; then
    echo "data/embeddings not found: download the embeddings with 'bash data/download.sh'" >&2
    exit 1
  fi
  bash experiments/scripts/get_driftlens.sh
}

# Runs "SCRIPT --uc <use cases> ARGS..." in JOBS parallel processes. The use cases are dealt round-robin in the given
# order (order them so that the processes get a similar load). One log per process: LOG_DIR/NAME_<i>.log. Fails if
# any process fails.
# usage: in_parallel NAME SCRIPT "UC1 UC2 ..." [ARGS...]
in_parallel() {
  local name=$1 script=$2
  local ucs=($3)
  shift 3
  local pids=() q i
  for ((q = 0; q < JOBS; q++)); do
    local mine=()
    for ((i = q; i < ${#ucs[@]}; i += JOBS)); do mine+=("${ucs[i]}"); done
    if [ ${#mine[@]} -eq 0 ]; then continue; fi
    log "  process $q: use cases ${mine[*]} (log: $LOG_DIR/${name}_$q.log)"
    py "$script" --uc "${mine[@]}" "$@" > "$LOG_DIR/${name}_$q.log" 2>&1 &
    pids+=($!)
  done
  local status=0
  for q in "${pids[@]}"; do wait "$q" || status=1; done
  if [ $status -ne 0 ]; then
    echo "$name: at least one process failed, see $LOG_DIR/${name}_*.log" >&2
  fi
  return $status
}
