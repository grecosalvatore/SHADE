""" Figures of the paper from the results in results/raw.

    timing.pdf               execution time per window vs window size, DriftLens and SHADE (K = 64, B = 2)
    drift_stream.pdf         global score / threshold over streams with sudden, incremental and periodic drift
    drift_stream_labels.pdf  per-label score / per-label threshold over the same streams

A figure is skipped when its results are missing.

Example:
    python experiments/make_figures.py
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from use_cases import RAW_DIR, RESULTS_DIR, load_json

DRIFTLENS, SHADE = "DriftLens", "SHADE_K64_B2"
GREEN, BLUE, GRID = "#1baf7a", "#2a78d6", "#e4e3df"
# detector: (label, color, marker, line style)
STYLE = {DRIFTLENS: ("DriftLens", GREEN, "^", "-"), SHADE: ("SHADE", BLUE, "o", "-")}
THRESHOLD_LINE = dict(color="#555555", linewidth=0.7, linestyle=(0, (3, 2)))
DRIFT_COLOR = "#8c8c8c"
LABEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Class names of the use cases shown in the per-label figure; other use cases are shown as "class <id>"
LABEL_NAMES = {"01": {"0": "World", "1": "Sports", "2": "Business"},
               "06": {"0": "Forest", "1": "Building", "2": "Glacier", "3": "Street", "4": "Mountain"}}
PATTERNS = ("sudden", "incremental", "periodic")
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
      "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
      "legend.fontsize": 7, "legend.frameon": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
      "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.color": "#222222", "ytick.color": "#222222",
      "axes.edgecolor": "#222222", "axes.labelcolor": "#000000", "axes.spines.top": False, "axes.spines.right": False,
      "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.5, "lines.markersize": 5,
      "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42}


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metadata = {"CreationDate": None} if path.endswith(".pdf") else None  # same file at every run
    fig.savefig(path, metadata=metadata)
    plt.close(fig)
    print(f"  {path}")
    return


def figure_timing(t, path):
    """ Time per window vs window size: mean (line) and standard deviation (band) over the runs. """
    sizes = np.array(t["sizes"])
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for det in (DRIFTLENS, SHADE):
        label, color, marker, ls = STYLE[det]
        y, e = map(np.array, zip(*[t["summary"][det][str(s)] for s in sizes]))
        ax.fill_between(sizes, y - e, y + e, color=color, alpha=0.18, linewidth=0)
        ax.plot(sizes, y, color=color, marker=marker, linestyle=ls, linewidth=1.2, markersize=3.2, markeredgewidth=0, label=label)
    ax.set_xlim(0, sizes[-1] * 1.02)
    ax.set_ylim(0, None)
    ax.set_xticks(np.arange(0, sizes[-1] + 1, 1000 if sizes[-1] > 2000 else 250))
    if len(sizes) > 1:
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(sizes[1] - sizes[0]))
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.4)
    ax.set_xlabel("Window size $m_w$ (samples)")
    ax.set_ylabel("Time per window (ms)")
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.grid(False, axis="x")
    ax.legend(loc="upper left", frameon=False, handlelength=1.8)
    fig.tight_layout(pad=0.3)
    save(fig, path)
    return


def drift_background(ax, drift):
    """ Grey background on the windows with injected drift (darker = more drift). """
    p_max = max(drift.max(), 1e-9)
    for k in np.flatnonzero(drift > 0):
        ax.axvspan(k - 0.5, k + 0.5, color=DRIFT_COLOR, alpha=0.08 + 0.27 * drift[k] / p_max, linewidth=0)
    return


def ratio_axes(ax, n_windows):
    """ Threshold line at 1 and logarithmic axis for the score / threshold ratio. """
    ax.axhline(1, **THRESHOLD_LINE)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(-0.5, n_windows - 0.5)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.grid(False, axis="x")
    return


def max_drift(streams):
    return 100 * max(max(r["patterns"][name]["drift"]) for r in streams for name in PATTERNS)


def figure_drift_stream(streams, path):
    """ Global score / threshold over the stream: one row per use case, one column per drift pattern. """
    fig, axes = plt.subplots(len(streams), len(PATTERNS), figsize=(7.16, 1.5 * len(streams) + 0.5), sharex=True,
                             sharey="row", squeeze=False)
    for i, r in enumerate(streams):
        for j, name in enumerate(PATTERNS):
            ax, pat = axes[i, j], r["patterns"][name]
            drift = np.array(pat["drift"])
            drift_background(ax, drift)
            for det in (DRIFTLENS, SHADE):
                label, color, _, ls = STYLE[det]
                ax.plot(np.arange(len(drift)), np.array(pat["scores"][det]) / r["threshold"][det], color=color, linestyle=ls,
                        linewidth=1.0, label=label)
            ratio_axes(ax, len(drift))
            if i == 0:
                ax.set_title(f"{name.capitalize()} drift", fontsize=8, pad=4)
            if j == 0:
                ax.set_ylabel(f"UC{r['uc']} {r['dataset']}\nScore / threshold", fontsize=7)
            if i == len(streams) - 1:
                ax.set_xlabel("Window")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], **THRESHOLD_LINE), Patch(color=DRIFT_COLOR, alpha=0.3, linewidth=0)]
    labels += ["threshold", f"injected drift (up to {max_drift(streams):.0f}%)"]
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.95))
    save(fig, path)
    return


def figure_drift_stream_labels(streams, path):
    """ Per-label score / per-label threshold over the stream: one row per (use case, detector), one column per drift
    pattern, one line per predicted label. """
    rows = [(r, det) for r in streams for det in (DRIFTLENS, SHADE)]
    fig, axes = plt.subplots(len(rows), len(PATTERNS), figsize=(7.16, 1.3 * len(rows) + 0.5), sharex=True, sharey="row",
                             squeeze=False)
    for i, (r, det) in enumerate(rows):
        for j, name in enumerate(PATTERNS):
            ax, pat = axes[i, j], r["patterns"][name]
            drift = np.array(pat["drift"])
            drift_background(ax, drift)
            for c, l in enumerate(r["labels"]):
                y = np.array(pat["scores_per_label"][det][l], dtype=float) / r["threshold_per_label"][det][l]
                ax.plot(np.arange(len(drift)), y, color=LABEL_COLORS[c % len(LABEL_COLORS)], linewidth=0.9,
                        label=LABEL_NAMES.get(r["uc"], {}).get(l, f"class {l}"))
            ratio_axes(ax, len(drift))
            if i == 0:
                ax.set_title(f"{name.capitalize()} drift", fontsize=8, pad=4)
            if j == 0:
                ax.set_ylabel(f"UC{r['uc']} {STYLE[det][0]}\nlabel score / threshold", fontsize=7)
            if i == len(rows) - 1:
                ax.set_xlabel("Window")
        if det == DRIFTLENS:
            axes[i, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=6.5,
                               title=f"UC{r['uc']} predicted label", title_fontsize=6.5, handlelength=1.4)
    handles = [plt.Line2D([], [], **THRESHOLD_LINE), Patch(color=DRIFT_COLOR, alpha=0.3, linewidth=0)]
    fig.legend(handles, ["threshold of each label", f"injected drift (up to {max_drift(streams):.0f}%)"], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.45, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.96))
    save(fig, path)
    return


def main():
    parser = argparse.ArgumentParser(description="Figures of the paper from the results in results/raw")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "figures"))
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()

    timing_path = os.path.join(args.raw_dir, "timing", "UC01.json")
    streams = [load_json(f) for f in sorted(glob.glob(os.path.join(args.raw_dir, "drift_stream", "UC*_mw*.json")))]
    print("Figures:")
    with plt.rc_context(RC):
        if os.path.exists(timing_path):
            figure_timing(load_json(timing_path), os.path.join(args.out_dir, f"timing.{args.format}"))
        if streams:
            figure_drift_stream(streams, os.path.join(args.out_dir, f"drift_stream.{args.format}"))
            figure_drift_stream_labels(streams, os.path.join(args.out_dir, f"drift_stream_labels.{args.format}"))
    return


if __name__ == "__main__":
    main()
