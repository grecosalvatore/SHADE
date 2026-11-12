""" Drift patterns over a simulated data stream: the data of the qualitative figures.

For each use case, a stream of --n-windows windows of --window-size samples is scored by DriftLens and SHADE
(K = 64, B = 2), globally and per predicted label, for three drift patterns:
    sudden       no drift until 40 % of the stream, then --max-drift
    incremental  linear ramp from 0 to --max-drift between 30 % and 70 % of the stream, then constant
    periodic     --max-drift in blocks of 10 % of the stream every 25 %, starting at 15 %
A window with drift percentage p contains (1 - p) m_w samples of the new unseen data and p m_w drifted samples. Both
detectors score the same windows; the sampler of the i-th pattern has seed 1000 r + 200 + i.

The detectors are fitted with the seeds of run --run of run_detection.py and score the --n-th threshold windows of
that run (sampler seed 1000 r): the global threshold and the threshold of each label are the 99th percentiles of the
global and per-label scores.

Output: <out-dir>/UC<id>_mw<m_w>.json with the drift percentage, the global and per-label scores of every window of
every pattern, and the thresholds. Existing outputs are skipped unless --overwrite is given.

Example:
    python experiments/run_drift_stream.py --uc 01 06
"""
import argparse
import os

import numpy as np

from baselines import RobustDriftLens
from metrics import T_ALPHA, threshold
from run_detection import SHADE_DEFAULT, build_detectors, fit_detectors
from sampling import stream_sampler, threshold_sampler
from use_cases import ALL_USE_CASES, EMBEDDINGS_DIR, RAW_DIR, UseCase, save_json

DETECTORS = ["DriftLens", SHADE_DEFAULT]


def patterns(n, max_drift):
    """ Drift percentage of each window of a stream of n windows, for each pattern. """
    t = np.arange(n)
    periodic = np.zeros(n)
    for start in range(int(0.15 * n), n, max(int(0.25 * n), 1)):
        periodic[start:start + max(int(0.10 * n), 1)] = max_drift
    return {"sudden": np.where(t >= int(0.4 * n), max_drift, 0.0),
            "incremental": max_drift * np.clip((t - 0.3 * n) / (0.4 * n), 0, 1),
            "periodic": periodic}


def score_stream(detectors, sampler, m_w, drift, labels):
    """ Global and per-label scores of every detector on a stream of windows with the given drift percentages:
    ({detector: [score per window]}, {detector: {label: [score per window]}}). """
    scores = {name: [] for name in detectors}
    scores_per_label = {name: {str(l): [] for l in labels} for name in detectors}
    for p in drift:
        E_w, Y_w = sampler.window(m_w, float(p))
        for name, det in detectors.items():
            s = det.score(E_w, Y_w)
            scores[name].append(float(s["global"]))
            for l in labels:
                scores_per_label[name][str(l)].append(float(s["per_label"][l]))
    return scores, scores_per_label


def main():
    parser = argparse.ArgumentParser(description="Scores of DriftLens and SHADE over streams with sudden, incremental and periodic drift")
    parser.add_argument("--uc", nargs="+", default=["01", "06"], choices=ALL_USE_CASES, metavar="UC")
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--n-windows", type=int, default=100, help="Windows per stream")
    parser.add_argument("--max-drift", type=float, default=0.20)
    parser.add_argument("--run", type=int, default=0, help="Run whose seeds are used")
    parser.add_argument("--n-th", type=int, default=10000, help="Threshold windows")
    parser.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--out-dir", default=os.path.join(RAW_DIR, "drift_stream"))
    parser.add_argument("--overwrite", action="store_true", help="Recompute the outputs that already exist")
    args = parser.parse_args()
    m_w, run = args.window_size, args.run

    for uc_id in args.uc:
        out_path = os.path.join(args.out_dir, f"UC{uc_id}_mw{m_w}.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"UC{uc_id}: {out_path} exists, skipped", flush=True)
            continue
        uc = UseCase(uc_id, args.embeddings_dir)
        detectors = build_detectors(uc, run, DETECTORS)
        fit_detectors(detectors, uc)
        detectors["DriftLens"] = RobustDriftLens(detectors["DriftLens"])

        # Thresholds: global and per label, from the threshold windows of the run
        th, th_per_label = score_stream(detectors, threshold_sampler(uc, run), m_w, np.zeros(args.n_th), uc.labels)
        tau = {name: threshold(th[name], T_ALPHA) for name in detectors}
        tau_per_label = {name: {l: threshold(s, T_ALPHA) for l, s in th_per_label[name].items()} for name in detectors}
        print(f"UC{uc_id} thresholds: " + ", ".join(f"{name} {tau[name]:.4g}" for name in detectors), flush=True)

        out = {"uc": uc_id, "dataset": uc.dataset, "m_w": m_w, "run": run, "T_alpha": T_ALPHA, "n_th": args.n_th,
               "labels": [str(l) for l in uc.labels], "threshold": tau, "threshold_per_label": tau_per_label, "patterns": {}}
        for i, (name, drift) in enumerate(patterns(args.n_windows, args.max_drift).items()):
            scores, scores_per_label = score_stream(detectors, stream_sampler(uc, run, i), m_w, drift, uc.labels)
            out["patterns"][name] = {"drift": drift.tolist(), "scores": scores, "scores_per_label": scores_per_label}
            print(f"UC{uc_id} {name}: " + ", ".join(
                f"{det} false alarms {np.mean(np.array(scores[det])[drift == 0] > tau[det]):.2f}, "
                f"detected {np.mean(np.array(scores[det])[drift > 0] > tau[det]):.2f}" for det in detectors), flush=True)
        save_json(out_path, out)
        print(f"UC{uc_id}  -> {out_path}", flush=True)
    return


if __name__ == "__main__":
    main()
