""" Tables of the paper, in LaTeX and Markdown, from the results in results/raw.

    main.tex, main.md                       detection per use case and window size, DriftLens vs SHADE (K = 64, B = 2)
    variants.tex, variants.md               SHADE with K in {64, 128} and B in {2, 8}, mean over all use cases and sizes
    drift_severity.tex, drift_severity.md   correlation between drift percentage and score, and onset of the alarms

A table is skipped when its results are missing.

Example:
    python experiments/make_tables.py
"""
import argparse
import glob
import os

import numpy as np

from metrics import onset
from use_cases import RAW_DIR, RESULTS_DIR, USE_CASES, load_json

DRIFTLENS, SHADE = "DriftLens", "SHADE_K64_B2"
VARIANTS = [("SHADE_K64_B2", 64, 2), ("SHADE_K64_B8", 64, 8), ("SHADE_K128_B2", 128, 2), ("SHADE_K128_B8", 128, 8)]
VOLUME_GROUPS = [("(a) Larger data volumes", ["01", "04", "05", "09", "10"]),
                 ("(b) Smaller data volumes", ["02", "03", "06", "07", "08"])]
HIGHLIGHT_TEXT = {"shadebetter": "higher", "tie": "equal", None: "lower"}  # row highlight, in words for the Markdown table

# LaTeX preamble required by the tables, written as a comment at the top of every .tex file
PREAMBLE = r"""% Required in the preamble:
%   \usepackage{booktabs}
%   \usepackage{graphicx}
%   \usepackage{amsmath}
%   \usepackage[table]{xcolor}
%   \definecolor{shadebetter}{HTML}{C6DBEF}
%   \definecolor{tie}{HTML}{EBEBEB}
%   \newcommand{\nocol}{\cellcolor{white}}
%   \newcommand{\method}{SHADE}
"""


def metric(x, key):
    """ One metric of one run of one detector. """
    if key in ("auc5", "auc10"):
        return x["auc"]["0.05" if key == "auc5" else "0.1"]
    if key in ("acc0", "acc5", "acc10"):
        return x["accuracy"][{"acc0": "0.0", "acc5": "0.05", "acc10": "0.1"}[key]]
    if key == "ms":
        return x["online_ms_median"]
    return x[key]


def run_values(r, det, key):
    """ Values of a metric of a detector over the runs of one (use case, window size). """
    return [metric(run[det], key) for run in r["runs"]]


def mean_over(results, det, key):
    """ Mean over the (use case, window size) pairs of each run, then [mean, std] over the runs. """
    n_runs = min(r["n_runs"] for r in results)
    per_run = [np.mean([metric(r["runs"][i][det], key) for r in results]) for i in range(n_runs)]
    return [float(np.mean(per_run)), float(np.std(per_run))]


def printed(value, digits):
    """ A value at the precision it is printed with: bold and highlights compare the printed values. """
    return float(f"{value:.{digits}f}")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"  {path}")
    return


# ----------------------------------------------------------------------------------------------------------------------
# Main table
# ----------------------------------------------------------------------------------------------------------------------
def table_main(results, out_dir):
    """ One row per use case and window size, grouped by data volume: Acc@Q, H_DD and AUC@5% of DriftLens and SHADE
    (mean and std over the runs). The better H_DD and AUC of each row are in bold, both on a tie. Rows are highlighted
    when the H_DD of SHADE is higher than (shadebetter) or equal to (tie) the one of DriftLens. The last row is the
    average over all rows. """
    by_uc = {}
    for r in results:
        by_uc.setdefault(r["uc"], {})[r["m_w"]] = r
    keys = [("acc0", "Acc.@0\\%", "Acc@0%"), ("acc5", "Acc.@5\\%", "Acc@5%"), ("acc10", "Acc.@10\\%", "Acc@10%"),
            ("hdd", "H$_{\\text{DD}}$", "H_DD"), ("auc5", "AUC@5\\%", "AUC@5%")]
    detectors = [DRIFTLENS, SHADE]
    nk = len(keys)

    def cells(stats, tex):
        """ stats: {detector: {key: [mean, std]}} """
        out = []
        for det in detectors:
            for k, _, _ in keys:
                mean, std = stats[det][k]
                bold = k in ("hdd", "auc5") and printed(mean, 2) >= max(printed(stats[d][k][0], 2) for d in detectors)
                m, s = f"{mean:.2f}", f"{std:.2f}"[1:]  # the std is below 1: ".01"
                if tex:
                    out.append((f"\\textbf{{{m}}}" if bold else m) + f"$_{{\\pm{s}}}$")
                else:
                    out.append((f"**{m}**" if bold else m) + f" ± {s}")
        return out

    def highlight(stats):
        shade, driftlens = printed(stats[SHADE]["hdd"][0], 2), printed(stats[DRIFTLENS]["hdd"][0], 2)
        return "shadebetter" if shade > driftlens else "tie" if shade == driftlens else None

    tex = ["\\begin{table*}[htbp!]", "\\centering",
           "\\caption{\\textbf{Drift detection performance comparison.} Accuracy Acc@$Q\\%$ at drift percentage $Q$, "
           "H$_{\\text{DD}}$, and AUC at $Q=5\\%$ of \\textsc{DriftLens} and \\method{} for each use case and window size "
           "$m_w$. In each run, the metrics are computed on 1000 windows per drift level; values are the mean and standard "
           "deviation over 5 runs. The better H$_{\\text{DD}}$ and AUC of each row are in bold (both in case of a tie). "
           "Rows where \\method{} achieves a higher H$_{\\text{DD}}$ than \\textsc{DriftLens} are highlighted in "
           "\\colorbox{shadebetter}{blue}, rows where they are equal in \\colorbox{tie}{gray}. The last row reports the "
           "average over all rows.}",
           "\\label{tab:Performance_per_UC_new}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llr|" + "c" * nk + "|" + "c" * nk + "}", "\\toprule",
           f" & & & \\multicolumn{{{nk}}}{{c}}{{\\textsc{{DriftLens}}}} & \\multicolumn{{{nk}}}{{c}}{{\\textsc{{\\method{{}}}}}} \\\\",
           f"\\cmidrule(lr){{4-{3 + nk}}} \\cmidrule(lr){{{4 + nk}-{3 + 2 * nk}}}",
           "\\textbf{Use case} & \\textbf{Model} & $m_w$ & " + " & ".join(h for _ in detectors for _, h, _ in keys) + " \\\\"]
    md = ["# Drift detection performance: DriftLens vs SHADE (K = 64, B = 2)\n",
          "Acc@Q: accuracy on the windows with drift percentage Q; H_DD; AUC at Q = 5%. Mean ± standard deviation over "
          "5 runs, 1000 windows per drift percentage in each run. The better H_DD and AUC of each row are in bold (both "
          "on a tie). Last column: whether the H_DD of SHADE is higher than, equal to or lower than the one of DriftLens. The last row is the average over all rows.\n",
          "| Use case | Model | m_w | " + " | ".join(f"{name} {h}" for name in ("DriftLens", "SHADE") for _, _, h in keys)
          + " | SHADE vs DriftLens (H_DD) |", "|---" * (4 + 2 * nk) + "|"]
    for title, group in VOLUME_GROUPS:
        group = [uc for uc in group if uc in by_uc]
        if not group:
            continue
        tex += ["\\midrule", f"\\multicolumn{{{3 + 2 * nk}}}{{l}}{{\\textit{{{title}}}}} \\\\", "\\midrule"]
        md.append(f"| **{title}** |" + " |" * (3 + 2 * nk))
        for j, uc in enumerate(group):
            # The use case name takes the first row; a parenthesis in the name, e.g. "(motors)", goes to the second row
            base, _, extra = by_uc[uc][min(by_uc[uc])]["dataset"].partition(" (")
            names = [f"UC{uc} {base}"] + ([f"({extra}"] if extra else [])
            for i, m_w in enumerate(sorted(by_uc[uc])):
                r = by_uc[uc][m_w]
                stats = {det: {k: [float(np.mean(run_values(r, det, k))), float(np.std(run_values(r, det, k)))]
                               for k, _, _ in keys} for det in detectors}
                name = names[i] if i < len(names) else ""
                model = USE_CASES[uc]["model"] if i == 0 else ""
                tex_name = f"\\hspace{{1em}}{name}" if i > 0 and name else name
                color = highlight(stats)
                if color:  # the first two cells stay white
                    first = f"\\rowcolor{{{color}}} \\nocol{' ' + tex_name if tex_name else ''} & \\nocol{' ' + model if model else ''}"
                else:
                    first = f"{tex_name} & {model}"
                tex.append(f"{first} & {m_w} & " + " & ".join(cells(stats, True)) + " \\\\")
                md.append(f"| {name} | {model} | {m_w} | " + " | ".join(cells(stats, False))
                          + f" | {HIGHLIGHT_TEXT[color]} |")
            if j < len(group) - 1:
                tex.append("\\addlinespace")
    average = {det: {k: mean_over(results, det, k) for k, _, _ in keys} for det in detectors}
    tex += ["\\midrule", "\\multicolumn{3}{l}{\\textbf{Average}} & " + " & ".join(cells(average, True)) + " \\\\",
            "\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    md.append("| **Average** | | | " + " | ".join(cells(average, False)) + " | |")
    write(os.path.join(out_dir, "main.tex"), PREAMBLE + "\n".join(tex) + "\n")
    write(os.path.join(out_dir, "main.md"), "\n".join(md) + "\n")
    return


# ----------------------------------------------------------------------------------------------------------------------
# Variants table
# ----------------------------------------------------------------------------------------------------------------------
def table_variants(results, out_dir):
    """ SHADE with K directions and B bins, mean over all (use case, window size) pairs, with DriftLens as reference.
    The best SHADE value of each column is in bold. """
    # (key, LaTeX header, Markdown header, +1 if higher is better, decimals)
    cols = [("auc5", "AUC@5\\%", "AUC@5%", 1, 3), ("auc10", "AUC@10\\%", "AUC@10%", 1, 3), ("acc0", "Acc.@0\\%", "Acc@0%", 1, 3),
            ("hdd", "H$_{\\text{DD}}$", "H_DD", 1, 3), ("seed", "Seed std", "Seed std", -1, 3), ("ms", "ms/window", "ms/window", -1, 2),
            ("kb", "Stored (kB)", "Stored (kB)", -1, 1)]
    one_per_uc = list({r["uc"]: r for r in results}.values())

    def row_values(det):
        v = {k: mean_over(results, det, k)[0] for k in ("auc5", "auc10", "acc0", "hdd", "ms")}
        # Seed std: std of AUC@5% over the runs of each (use case, window size), averaged
        v["seed"] = float(np.mean([np.std(run_values(r, det, "auc5")) for r in results]))
        # Stored state: mean over use cases (it does not depend on the window size)
        v["kb"] = float(np.mean([r["summary"][det]["nbytes"] for r in one_per_uc]) / 1e3)
        return v

    variants = [v for v in VARIANTS if all(v[0] in r["summary"] for r in results)]
    vals = {det: row_values(det) for det in [DRIFTLENS] + [v[0] for v in variants]}
    best = {k: (max if sign > 0 else min)(printed(vals[det][k], d) for det, _, _ in variants) for k, _, _, sign, d in cols}

    tex = ["\\begin{table*}[htbp!]", "\\centering",
           "\\caption{\\textbf{Ablation study and efficiency evaluation.} Comparison of \\method variants with $K$ random "
           "directions and $B$ bins per direction, mean over all use cases and window sizes (5 runs each). Seed std: "
           "standard deviation of AUC@5\\% over the runs of each use case and window size, averaged. Stored: reference "
           "state, mean over use cases. Best \\method{} value per column in bold; DriftLens is reported as reference}",
           "\\label{tab:variants_new}", "\\setlength{\\tabcolsep}{8pt}", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
           "\\textbf{Variant} & " + " & ".join(h for _, h, _, _, _ in cols) + " \\\\", "\\midrule"]
    md = ["# SHADE variants: K random directions, B bins per direction\n",
          "Mean over all use cases and window sizes (5 runs each). Seed std: standard deviation of AUC@5% over the runs of "
          "each use case and window size, averaged. Stored: reference state, mean over use cases. ms/window: median time "
          "per window measured during the detection runs. Best SHADE value per column in bold; DriftLens as reference.\n",
          "| Variant | " + " | ".join(h for _, _, h, _, _ in cols) + " |", "|---" * (1 + len(cols)) + "|"]
    for det, K, B in variants:
        cells = [(f"{vals[det][k]:.{d}f}", printed(vals[det][k], d) == best[k]) for k, _, _, _, d in cols]
        default = " (default)" if det == SHADE else ""
        tex.append(f"$K{{=}}{K},\\ B{{=}}{B}${default} & " + " & ".join(f"\\textbf{{{c}}}" if b else c for c, b in cells) + " \\\\")
        md.append(f"| K = {K}, B = {B}{default} | " + " | ".join(f"**{c}**" if b else c for c, b in cells) + " |")
    reference = [f"{vals[DRIFTLENS][k]:.{d}f}" for k, _, _, _, d in cols]
    tex += ["\\midrule", "DriftLens & " + " & ".join(reference) + " \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    md.append("| DriftLens | " + " | ".join(reference) + " |")
    write(os.path.join(out_dir, "variants.tex"), PREAMBLE + "\n".join(tex) + "\n")
    write(os.path.join(out_dir, "variants.md"), "\n".join(md) + "\n")
    return


# ----------------------------------------------------------------------------------------------------------------------
# Drift-severity table
# ----------------------------------------------------------------------------------------------------------------------
def table_drift_severity(sweep, out_dir):
    """ Spearman correlation between the drift percentage of a window and its score, over 0-100 % and 0-20 % drift,
    and onset (smallest drift percentage whose mean score exceeds the threshold), per use case, with the mean and the
    median over the use cases. The better value of each DriftLens / SHADE pair is in bold, both on a tie. """
    detectors = [DRIFTLENS, SHADE]
    sweep = sorted(sweep, key=lambda r: r["uc"])
    m_w = sweep[0]["m_w"]

    def rho_cell(v, tex):
        s = f"{v:.3f}"
        return s.replace("-", "$-$") if tex else s

    def onset_cell(v, tex):
        if v is None:
            return "--"
        return f"{100 * v:g}" + ("\\%" if tex else "%")

    def row(label, rho_all, rho_0_20, onsets, tex):
        """ rho_all, rho_0_20, onsets: {detector: value}; an onset of None is printed as '--' and never in bold. """
        best_all = max(printed(v, 3) for v in rho_all.values())
        best_0_20 = max(printed(v, 3) for v in rho_0_20.values())
        best_onset = min((v for v in onsets.values() if v is not None), default=None)
        bold = (lambda s: f"\\textbf{{{s}}}") if tex else (lambda s: f"**{s}**")
        cells = []
        for det in detectors:
            a, b, o = rho_cell(rho_all[det], tex), rho_cell(rho_0_20[det], tex), onset_cell(onsets[det], tex)
            cells += [bold(a) if printed(rho_all[det], 3) == best_all else a,
                      bold(b) if printed(rho_0_20[det], 3) == best_0_20 else b,
                      bold(o) if onsets[det] is not None and onsets[det] == best_onset else o]
        return (f"{label} & " + " & ".join(cells) + " \\\\") if tex else (f"| {label} | " + " | ".join(cells) + " |")

    rho_all = {r["uc"]: {det: r["summary"][det]["rho_all"][0] for det in detectors} for r in sweep}
    rho_0_20 = {r["uc"]: {det: r["summary"][det]["rho_0_20"][0] for det in detectors} for r in sweep}
    onsets = {r["uc"]: {det: onset(r["levels"], {p: v[0] for p, v in r["summary"][det]["mean_ratio"].items()})
                        for det in detectors} for r in sweep}
    ucs = [r["uc"] for r in sweep]
    aggregate = {}
    for name, f in (("Mean", np.mean), ("Median", np.median)):
        with_onset = name == "Median" and all(onsets[uc][det] is not None for uc in ucs for det in detectors)
        aggregate[name] = ({det: float(f([rho_all[uc][det] for uc in ucs])) for det in detectors},
                           {det: float(f([rho_0_20[uc][det] for uc in ucs])) for det in detectors},
                           {det: float(f([onsets[uc][det] for uc in ucs])) if with_onset else None for det in detectors})

    tex = ["\\begin{table}[t!]", "\\centering",
           "\\caption{\\textbf{Drift severity and score correlation.} Correlation of the drift score to the drift percentage "
           f"($m_w={m_w}$; mean over 5 runs). $\\rho_{{Q[0-100]}}$ and $\\rho_{{Q[0-20]}}$: Spearman correlation between the "
           "drift percentage of a window and its score, over windows with 0--100\\% and 0--20\\% drift (1: the score always "
           "increases with the drift). Onset: smallest drift percentage at which the mean score of the windows exceeds the "
           "threshold. Better value of each pair in bold}",
           "\\label{tab:drift_sweep}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lcccccc}", "\\toprule",
           " & \\multicolumn{3}{c}{\\textbf{DriftLens}} & \\multicolumn{3}{c}{\\textbf{\\method{}}} \\\\",
           "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
           "\\textbf{UC} & " + " & ".join("$\\rho_{Q[0-100]}$ & $\\rho_{Q[0-20]}$ & Onset" for _ in detectors) + " \\\\", "\\midrule"]
    md = [f"# Drift severity and score correlation (m_w = {m_w}, mean over 5 runs)\n",
          "rho: Spearman correlation between the drift percentage of a window and its score, over the windows with 0-100% "
          "and 0-20% drift. Onset: smallest drift percentage at which the mean score of the windows exceeds the threshold. "
          "Better value of each DriftLens / SHADE pair in bold.\n",
          "| UC | " + " | ".join(f"{name} rho 0-100% | {name} rho 0-20% | {name} onset" for name in ("DriftLens", "SHADE")) + " |",
          "|---" * 7 + "|"]
    for uc in ucs:
        tex.append(row(f"UC{uc}", rho_all[uc], rho_0_20[uc], onsets[uc], True))
        md.append(row(f"UC{uc}", rho_all[uc], rho_0_20[uc], onsets[uc], False))
    tex.append("\\midrule")
    for name, values in aggregate.items():
        tex.append(row(f"\\textbf{{{name}}}", *values, True))
        md.append(row(f"**{name}**", *values, False))
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(os.path.join(out_dir, "drift_severity.tex"), PREAMBLE + "\n".join(tex) + "\n")
    write(os.path.join(out_dir, "drift_severity.md"), "\n".join(md) + "\n")
    return


def main():
    parser = argparse.ArgumentParser(description="Tables of the paper from the results in results/raw")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "tables"))
    args = parser.parse_args()

    results = [load_json(f) for f in sorted(glob.glob(os.path.join(args.raw_dir, "detection", "UC*_mw*.json")))]
    sweep = [load_json(f) for f in sorted(glob.glob(os.path.join(args.raw_dir, "drift_sweep", "UC*_mw*.json")))]
    print(f"Tables from {len(results)} detection results and {len(sweep)} drift-sweep results:")
    if results:
        table_main(results, args.out_dir)
        table_variants(results, args.out_dir)
    if sweep:
        table_drift_severity(sweep, args.out_dir)
    return


if __name__ == "__main__":
    main()
