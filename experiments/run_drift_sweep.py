""" Drift score as a function of the drift percentage: the data of the drift-severity table.

For each use case, at one window size, DriftLens and SHADE (K = 64, B = 2) score windows with an increasing percentage
of drifted samples, from 0 % to 100 % in steps of 5 %: a window with drift percentage p contains (1 - p) m_w samples
of the new unseen data and p m_w drifted samples. For the use case with one drift file per drift percentage (UC08),
the drifted samples come from the file of the lowest percentage.

For each run r, the detectors are fitted with the seeds of run r of run_detection.py and their thresholds are
estimated on the same --n-th threshold windows (sampler seed 1000 r). Scores are divided by the threshold of the run,
so 1 is the alarm level of both detectors. Only the global score is computed: with a high percentage of drifted samples
a window can miss some predicted labels, where the per-label distances of DriftLens are not defined.

Output: <out-dir>/UC<id>_mw<m_w>.json with, for every run, the threshold, the mean score / threshold and the share
of flagged windows per drift percentage, and the Spearman correlation between drift percentage and score over 0-100 %
and over 0-20 %; and their mean and standard deviation over the runs. Existing outputs are skipped unless --overwrite
is given.

Example:
    python experiments/run_drift_sweep.py --uc 01 02
"""
import argparse
import os

import numpy as np

from baselines import GlobalDriftLens
from metrics import T_ALPHA, mean_std, spearman_rho, threshold
from run_detection import SHADE_DEFAULT, build_detectors, fit_detectors, score_windows
from sampling import sweep_samplers, threshold_sampler
from use_cases import ALL_USE_CASES, EMBEDDINGS_DIR, RAW_DIR, UseCase, save_json

DETECTORS = ["DriftLens", SHADE_DEFAULT]


def fitted_global_detectors(uc, run):
    """ DriftLens (global score only) and SHADE, fitted with the seeds of a run. """
    detectors = build_detectors(uc, run, DETECTORS)
    fit_detectors(detectors, uc)
    detectors["DriftLens"] = GlobalDriftLens(detectors["DriftLens"])
    return detectors


def main():
    parser = argparse.ArgumentParser(description="Drift score of DriftLens and SHADE vs drift percentage")
    parser.add_argument("--uc", nargs="+", default=ALL_USE_CASES, choices=ALL_USE_CASES, metavar="UC")
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--step", type=float, default=0.05, help="Step of the drift percentage, from 0 to 1")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--n-windows", type=int, default=200, help="Windows per drift percentage and run")
    parser.add_argument("--n-th", type=int, default=10000, help="Threshold windows per run")
    parser.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--out-dir", default=os.path.join(RAW_DIR, "drift_sweep"))
    parser.add_argument("--overwrite", action="store_true", help="Recompute the outputs that already exist")
    args = parser.parse_args()
    m_w = args.window_size
    levels = [round(float(p), 4) for p in np.arange(0, 1 + 1e-9, args.step)]

    for uc_id in args.uc:
        out_path = os.path.join(args.out_dir, f"UC{uc_id}_mw{m_w}.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"UC{uc_id}: {out_path} exists, skipped", flush=True)
            continue
        uc = UseCase(uc_id, args.embeddings_dir)
        runs = []
        for run in range(args.runs):
            detectors = fitted_global_detectors(uc, run)
            th, _ = score_windows(detectors, threshold_sampler(uc, run), args.n_th, m_w, 0.0)
            tau = {name: threshold(th[name], T_ALPHA) for name in detectors}
            print(f"UC{uc_id} run {run} thresholds: " + ", ".join(f"{name} {tau[name]:.4g}" for name in detectors), flush=True)
            ratio = {name: {} for name in detectors}
            for p, (sampler, injected) in sweep_samplers(uc, run, levels).items():
                scores, _ = score_windows(detectors, sampler, args.n_windows, m_w, injected)
                for name in detectors:
                    ratio[name][p] = scores[name] / tau[name]
            runs.append({name: {"threshold": tau[name],
                                "mean_ratio": {str(p): float(np.nanmean(s)) for p, s in ratio[name].items()},
                                "detected": {str(p): float(np.nanmean(s > 1)) for p, s in ratio[name].items()},
                                "rho_all": spearman_rho(ratio[name]), "rho_0_20": spearman_rho(ratio[name], 0.20)}
                         for name in detectors})
        summary = {name: {"mean_ratio": {str(p): mean_std([r[name]["mean_ratio"][str(p)] for r in runs]) for p in levels},
                          "detected": {str(p): mean_std([r[name]["detected"][str(p)] for r in runs]) for p in levels},
                          "rho_all": mean_std([r[name]["rho_all"] for r in runs]),
                          "rho_0_20": mean_std([r[name]["rho_0_20"] for r in runs])} for name in DETECTORS}
        save_json(out_path, {"uc": uc_id, "dataset": uc.dataset, "m_w": m_w, "levels": levels, "n_runs": args.runs,
                             "n_th": args.n_th, "n_windows": args.n_windows, "T_alpha": T_ALPHA, "runs": runs, "summary": summary})
        print(f"UC{uc_id}: " + ", ".join(f"{name} rho {summary[name]['rho_all'][0]:.3f} (0-20%: {summary[name]['rho_0_20'][0]:.3f})"
                                         for name in DETECTORS) + f"  -> {out_path}", flush=True)
    return


if __name__ == "__main__":
    main()
