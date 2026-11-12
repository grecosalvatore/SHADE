""" Drift detection of DriftLens and SHADE (K in {64, 128} directions, B in {2, 8} bins) for each use case and window
size: the data of the main table and of the variants table.

For each (use case, window size) and run r:
  1. every detector is fitted on the training embeddings;
  2. the threshold of each detector is the 99th percentile of its scores on --n-th drift-free windows of the test set;
  3. every detector scores --n-windows windows of the new unseen data per drift percentage (0 %, 5 %, 10 %).
All detectors score the same windows. Run r uses the sampler seeds 1000 r + {0, 1, 2} (threshold, no drift, drift) and
the seed r for the SHADE directions. A window size is skipped when a class-balanced window does not fit in the
smallest label pool (windows are sampled without repetition).

Output: <out-dir>/UC<id>_mw<m_w>.json with the metrics, times and stored-state size of every run, and their mean and
standard deviation over the runs. Existing outputs are skipped unless --overwrite is given.

Example:
    python experiments/run_detection.py --uc 07 --window-sizes 250
"""
import argparse
import os
import time

import numpy as np

from shade import SHADE

from baselines import DriftLensDetector
from metrics import T_ALPHA, evaluate, mean_std
from sampling import DRIFT_PERCENTAGES, detection_samplers, threshold_sampler
from use_cases import ALL_USE_CASES, EMBEDDINGS_DIR, RAW_DIR, UseCase, save_json

WINDOW_SIZES = [250, 500, 1000, 2000]
SHADE_VARIANTS = [(64, 2), (64, 8), (128, 2), (128, 8)]  # (K, B); the first one is the default of the paper


def shade_name(K, B):
    return f"SHADE_K{K}_B{B}"


SHADE_DEFAULT = shade_name(*SHADE_VARIANTS[0])
DETECTORS = ["DriftLens"] + [shade_name(K, B) for K, B in SHADE_VARIANTS]


def build_detectors(uc, run, names=DETECTORS):
    """ Detectors of a run, {name: detector}. The SHADE directions are seeded with the run index. """
    detectors = {}
    for name in names:
        if name == "DriftLens":
            detectors[name] = DriftLensDetector(uc.d1, uc.d1_l)
        else:
            K, B = next(v for v in SHADE_VARIANTS if shade_name(*v) == name)
            detectors[name] = SHADE(n_directions=K, n_bins=B, seed=run)
    return detectors


def fit_detectors(detectors, uc):
    """ Fits every detector on the training embeddings. Returns the fit time in seconds of each one. """
    fit_time = {}
    for name, det in detectors.items():
        start_time = time.perf_counter()
        det.fit(uc.E_train, uc.Yp_train, uc.labels)
        fit_time[name] = time.perf_counter() - start_time
    return fit_time


def score_windows(detectors, sampler, n, window_size, drift_percentage):
    """ Global score of every detector on the same n windows, and time in seconds of each scoring call (global and
    per-label scores; the sampling of the window is not timed). """
    scores = {name: np.empty(n) for name in detectors}
    elapsed = {name: np.empty(n) for name in detectors}
    for i in range(n):
        E_w, Y_w = sampler.window(window_size, drift_percentage)
        for name, det in detectors.items():
            start_time = time.perf_counter()
            s = det.score(E_w, Y_w)
            elapsed[name][i] = time.perf_counter() - start_time
            scores[name][i] = s["global"]
    return scores, elapsed


def run_once(uc, m_w, run, args):
    """ Fits every detector, estimates its threshold and evaluates it, for one run. """
    detectors = build_detectors(uc, run, args.detectors)
    fit_time = fit_detectors(detectors, uc)
    res = {name: {"fit_time_s": fit_time[name], "nbytes": det.state_nbytes()} for name, det in detectors.items()}

    th, th_elapsed = score_windows(detectors, threshold_sampler(uc, run), args.n_th, m_w, 0.0)
    online, elapsed = {}, {name: [] for name in detectors}
    for p, (sampler, injected) in detection_samplers(uc, run).items():
        online[p], el = score_windows(detectors, sampler, args.n_windows, m_w, injected)
        for name in detectors:
            elapsed[name].append(el[name])

    for name in detectors:
        res[name].update(evaluate(th[name], {p: online[p][name] for p in online}, T_ALPHA))
        el = np.concatenate(elapsed[name]) * 1e3
        res[name]["online_ms_median"], res[name]["online_ms_mean"] = float(np.median(el)), float(el.mean())
        res[name]["threshold_time_s"] = float(th_elapsed[name].sum())
    return res


def summarize(runs):
    """ Mean and standard deviation over the runs of the metrics of each detector. """
    out = {}
    for name in runs[0]:
        r = [run[name] for run in runs]
        out[name] = {"auc5": mean_std([x["auc"]["0.05"] for x in r]), "auc10": mean_std([x["auc"]["0.1"] for x in r]),
                     "acc0": mean_std([x["accuracy"]["0.0"] for x in r]), "acc5": mean_std([x["accuracy"]["0.05"] for x in r]),
                     "acc10": mean_std([x["accuracy"]["0.1"] for x in r]), "far": mean_std([x["far"] for x in r]),
                     "hdd": mean_std([x["hdd"] for x in r]), "online_ms_median": mean_std([x["online_ms_median"] for x in r]),
                     "fit_time_s": mean_std([x["fit_time_s"] for x in r]), "nbytes": int(r[0]["nbytes"])}
    return out


def main():
    parser = argparse.ArgumentParser(description="Drift detection of DriftLens and SHADE per use case and window size")
    parser.add_argument("--uc", nargs="+", default=ALL_USE_CASES, choices=ALL_USE_CASES, metavar="UC")
    parser.add_argument("--window-sizes", type=int, nargs="+", default=WINDOW_SIZES)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--n-th", type=int, default=10000, help="Threshold windows per run")
    parser.add_argument("--n-windows", type=int, default=1000, help="Windows per drift percentage and run")
    parser.add_argument("--detectors", nargs="+", default=DETECTORS, choices=DETECTORS)
    parser.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--out-dir", default=os.path.join(RAW_DIR, "detection"))
    parser.add_argument("--overwrite", action="store_true", help="Recompute the outputs that already exist")
    args = parser.parse_args()

    for uc_id in args.uc:
        uc = None
        for m_w in sorted(args.window_sizes):
            out_path = os.path.join(args.out_dir, f"UC{uc_id}_mw{m_w}.json")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"UC{uc_id} m_w={m_w}: {out_path} exists, skipped", flush=True)
                continue
            uc = uc or UseCase(uc_id, args.embeddings_dir)
            pool_fraction = uc.pool_fraction(m_w)
            if pool_fraction > 1:
                print(f"UC{uc_id} m_w={m_w}: a window does not fit in the smallest label pool, skipped", flush=True)
                continue
            runs = []
            for run in range(args.runs):
                start_time = time.perf_counter()
                runs.append(run_once(uc, m_w, run, args))
                print(f"UC{uc_id} m_w={m_w} run {run}: {time.perf_counter() - start_time:.0f} s", flush=True)
            summary = summarize(runs)
            save_json(out_path, {"uc": uc_id, "dataset": uc.dataset, "m_w": m_w, "pool_fraction": pool_fraction,
                                 "n_runs": args.runs, "n_th": args.n_th, "n_windows": args.n_windows, "T_alpha": T_ALPHA,
                                 "drift_percentage": [0.0] + DRIFT_PERCENTAGES, "d1": uc.d1, "d1_l": uc.d1_l,
                                 "runs": runs, "summary": summary})
            print(f"UC{uc_id} m_w={m_w} ({args.runs} runs): " + "  ".join(
                f"{name} AUC@5% {v['auc5'][0]:.3f} H_DD {v['hdd'][0]:.3f} {v['online_ms_median'][0]:.2f} ms"
                for name, v in summary.items()) + f"  -> {out_path}", flush=True)
    return


if __name__ == "__main__":
    main()
