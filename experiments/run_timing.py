""" Execution time per window of DriftLens and SHADE (K = 64, B = 2) as a function of the window size: the data of the
execution-time figure.

For each run the detectors are fitted once, then each window size is timed on --n-windows random windows of the new
unseen data, after one warm-up window; the time of a run is the median over its windows. Only the scoring of a window
(global and per-label scores) is timed, not its sampling.

Run it alone on the machine, with single-threaded linear algebra (experiments/scripts/01_timing.sh does both).

Output: <out-dir>/UC<id>.json with the time in ms of every run and window size, and the mean and standard deviation
over the runs. An existing output is skipped unless --overwrite is given.

Example:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/run_timing.py
"""
import argparse
import os
import time

import numpy as np

from metrics import mean_std
from run_detection import SHADE_DEFAULT, build_detectors, fit_detectors
from use_cases import ALL_USE_CASES, EMBEDDINGS_DIR, RAW_DIR, UseCase, save_json

DETECTORS = ["DriftLens", SHADE_DEFAULT]
WINDOW_SIZES = list(range(250, 5001, 250))


def time_windows(detectors, uc, m_w, n_windows, rng):
    """ Median time in ms per window of each detector on n_windows random windows of size m_w. """
    replace = m_w > len(uc.E_new)
    elapsed = {name: np.empty(n_windows) for name in detectors}
    for i in range(n_windows + 1):
        idx = rng.choice(len(uc.E_new), m_w, replace=replace)
        E_w, Y_w = uc.E_new[idx], uc.Yp_new[idx]
        for name, det in detectors.items():
            start_time = time.perf_counter()
            det.score(E_w, Y_w)
            if i > 0:  # the first window is a warm-up
                elapsed[name][i - 1] = time.perf_counter() - start_time
    return {name: float(np.median(v) * 1e3) for name, v in elapsed.items()}


def main():
    parser = argparse.ArgumentParser(description="Execution time per window of DriftLens and SHADE vs window size")
    parser.add_argument("--uc", default="01", choices=ALL_USE_CASES, metavar="UC")
    parser.add_argument("--window-sizes", type=int, nargs="+", default=WINDOW_SIZES)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--n-windows", type=int, default=20, help="Timed windows per window size and run")
    parser.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--out-dir", default=os.path.join(RAW_DIR, "timing"))
    parser.add_argument("--overwrite", action="store_true", help="Recompute the output if it already exists")
    args = parser.parse_args()

    out_path = os.path.join(args.out_dir, f"UC{args.uc}.json")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"UC{args.uc}: {out_path} exists, skipped", flush=True)
        return
    uc = UseCase(args.uc, args.embeddings_dir)
    runs, fits = [], []
    for run in range(args.runs):
        rng = np.random.default_rng(run)
        detectors = build_detectors(uc, run, DETECTORS)
        fits.append(fit_detectors(detectors, uc))
        runs.append({name: {} for name in detectors})
        for m_w in args.window_sizes:
            for name, ms in time_windows(detectors, uc, m_w, args.n_windows, rng).items():
                runs[-1][name][str(m_w)] = ms
            print(f"run {run} m_w={m_w}: " + "  ".join(f"{name} {runs[-1][name][str(m_w)]:.2f} ms" for name in detectors), flush=True)

    save_json(out_path, {"uc": args.uc, "dataset": uc.dataset, "d1": uc.d1, "d1_l": uc.d1_l, "n_runs": args.runs,
                         "n_windows": args.n_windows, "n_reference": len(uc.E_train), "n_features": uc.n_features,
                         "sizes": args.window_sizes, "runs": runs, "fit_time_s": fits,
                         "summary": {name: {str(m): mean_std([r[name][str(m)] for r in runs]) for m in args.window_sizes} for name in DETECTORS},
                         "fit_summary": {name: mean_std([f[name] for f in fits]) for name in DETECTORS}})
    print(f"UC{args.uc}  -> {out_path}", flush=True)
    return


if __name__ == "__main__":
    main()
