# SHADE: Lightweight Unsupervised Drift Detection for Constrained Devices

SHADE detects distribution drift in the embeddings of a deep learning model, without labels, at a cost that fits
constrained devices. Each embedding is projected on $K$ random directions; on each direction the reference data are
split into $B$ equiprobable bins, and a window of new data is summarized by its bin counts. The drift score is a
grouped Anderson–Darling statistic between the bin frequencies of the window and of the reference, averaged over the
directions; a window is flagged when its score exceeds a threshold estimated on drift-free data.

With the default configuration ($K = 64$, $B = 2$), SHADE stores about 7 kB of state (against about 2 MB for
DriftLens, the baseline of the paper), scores a window 4 to 12 times faster, and needs no PCA, covariance matrix or
matrix square root. The method is described in [shade/METHOD.md](shade/METHOD.md).

This repository contains:

| | |
|---|---|
| [shade/](shade/) | the method, as a small package that only needs NumPy |
| [examples/](examples/) | a quickstart on synthetic embeddings |
| [experiments/](experiments/) | the code of the experiments of the paper, with DriftLens as the baseline |
| [results/](results/) | the results of the paper: raw results, tables and figures |
| [data/](data/) | the download script and the description of the embeddings |

## Installation

```bash
git clone https://github.com/grecosalvatore/SHADE.git && cd SHADE
pip install -e .                    # the method only (NumPy)
pip install -e ".[experiments]"     # also what the experiments need, with the versions used in the paper
```

The method needs Python ≥ 3.9. The results of the paper were obtained with Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1,
scikit-learn 1.5.1, h5py 3.11.0 and matplotlib 3.9.2, which the `[experiments]` extra pins (use Python 3.9–3.12).

## Quick example

```python
import numpy as np
from shade import SHADE

shade = SHADE(n_directions=64, n_bins=2, seed=0)   # K = 64 directions, B = 2 bins (default)
shade.fit(E_ref, Y_ref)                            # reference embeddings (n, d) and their predicted labels

# Threshold: 99th percentile of the scores of drift-free windows
tau = np.quantile([shade.score(E_w)["global"] for E_w in drift_free_windows], 0.99)

scores = shade.score(E_w, Y_w)                     # {"global": score, "per_label": {label: score}}
drift = scores["global"] > tau
```

The labels are optional: without them SHADE returns the global score only. A complete, runnable example on synthetic
embeddings is in [examples/quickstart.py](examples/quickstart.py):

```bash
python examples/quickstart.py
```

## Reproducing the paper

There are two levels. Both need `pip install -e ".[experiments]"`.

### 1. From the stored results (about a minute)

The raw results of every experiment are in [results/raw/](results/raw/). The tables and figures of the paper are
built from them, without the embeddings:

```bash
python experiments/make_tables.py     # results/tables/
python experiments/make_figures.py    # results/figures/*.pdf
```

### 2. From the embeddings (about 14 hours with 2 processes on a laptop)

```bash
bash data/download.sh                                       # embeddings of the 10 use cases, 2.7 GB (see data/README.md)
RESULTS=results_rerun bash experiments/scripts/run_all.sh   # every experiment, then tables and figures
```

Every step skips the outputs that already exist, so the script can be stopped and started again. Since `results/`
already contains the results of the paper, `RESULTS=results_rerun` writes the recomputed results to a separate folder
that can be compared with `results/`. The steps can also be run one by one:

| Script | Experiment | Output in `$RESULTS/` | Time* |
|---|---|---|---|
| `01_timing.sh` | Time per window on UC01, window sizes 250 to 5000, single thread. Run it alone on the machine | `raw/timing/` | 5 min |
| `02_detection.sh` | Detection: 10 use cases × window sizes {250, 500, 1000, 2000} × 5 runs; DriftLens and SHADE with $K \in \lbrace 64, 128 \rbrace$, $B \in \lbrace 2, 8 \rbrace$ | `raw/detection/` | 11–12 h |
| `03_drift_sweep.sh` | Score vs drift percentage (0–100 % in steps of 5 %), window size 1000 | `raw/drift_sweep/` | 2 h |
| `04_drift_stream.sh` | Streams with sudden, incremental and periodic drift, UC01 and UC06 | `raw/drift_stream/` | 10 min |
| `05_tables_figures.sh` | Tables and figures | `tables/`, `figures/` | 1 min |

\*On an Apple M1 laptop with 16 GB of memory, 2 processes.

Optional environment variables: `PYTHON` (interpreter, default `python`), `JOBS` (parallel processes of steps 02 and
03, default 2), `NICE` (process priority, default 20, the lowest), `RESULTS` (results folder, default `results`).
Linear algebra is single-threaded in every step. The logs are in `$RESULTS/logs/`.

The Python scripts can also be called directly, for instance to recompute one use case and window size:

```bash
bash experiments/scripts/get_driftlens.sh    # DriftLens baseline (done automatically by the scripts above)
python experiments/run_detection.py --uc 07 --window-sizes 250 --out-dir results_rerun/raw/detection
```

**Baseline.** DriftLens is used at a fixed commit of its repository. It cannot be installed with pip, so
`experiments/scripts/get_driftlens.sh` downloads its source into `third_party/` and
[experiments/baselines.py](experiments/baselines.py) imports it from there.

**Determinism.** Windows and SHADE directions are seeded (run $r$: sampler seeds $1000r + \lbrace 0, 1, 2 \rbrace$, directions
seed $r$), so the SHADE results are reproduced exactly. The DriftLens results are reproduced up to numerical precision
on the use cases with a large training set; on the others (UC02, UC03, UC06, UC07, UC08) scikit-learn fits the PCA of
DriftLens with a randomized solver, and its thresholds and scores change slightly from one execution to the other
(on UC07 at window size 250, by up to about 0.01 in the metrics of a single run and 0.001 in their mean over the
5 runs). Execution times depend on the hardware.

## Experiments

| File | Content |
|---|---|
| [experiments/use_cases.py](experiments/use_cases.py) | the 10 use cases and the loading of their embeddings |
| [experiments/sampling.py](experiments/sampling.py) | class-balanced windows and drift injection |
| [experiments/baselines.py](experiments/baselines.py) | DriftLens adapters |
| [experiments/metrics.py](experiments/metrics.py) | Acc@Q, H_DD, AUC, Spearman correlation, onset |
| [experiments/run_detection.py](experiments/run_detection.py) | main table and variants table |
| [experiments/run_drift_sweep.py](experiments/run_drift_sweep.py) | drift-severity table |
| [experiments/run_drift_stream.py](experiments/run_drift_stream.py) | qualitative figures |
| [experiments/run_timing.py](experiments/run_timing.py) | execution-time figure |
| [experiments/make_tables.py](experiments/make_tables.py), [make_figures.py](experiments/make_figures.py) | `results/raw` → tables and figures |

**Protocol.** For each use case, window size $m_w$ and run: the detectors are fitted on the training embeddings; the
threshold of each detector is the 99th percentile of its scores on 10,000 drift-free windows of the test set; the
detectors then score 1000 windows of new data per drift percentage (0 %, 5 %, 10 %). Windows are class-balanced, and
all the detectors score the same windows. Results are the mean and standard deviation over 5 runs.

## Results

Average over the 10 use cases and the window sizes 250, 500, 1000 and 2000 (38 use case and window size pairs):

| | Acc@0% | Acc@5% | Acc@10% | H_DD | AUC@5% | ms per window | Stored state |
|---|---|---|---|---|---|---|---|
| DriftLens | 0.953 | 0.555 | 0.804 | 0.730 | 0.868 | 16.40 | 2048.4 kB |
| SHADE ($K = 64$, $B = 2$) | 0.918 | 0.544 | 0.785 | 0.683 | 0.841 | 2.48 | 7.1 kB |

| Paper | File | Raw results |
|---|---|---|
| Table: drift detection performance per use case and window size | [results/tables/main.md](results/tables/main.md) | `results/raw/detection/` |
| Table: SHADE variants ($K$, $B$) and efficiency | [results/tables/variants.md](results/tables/variants.md) | `results/raw/detection/` |
| Table: drift severity and score correlation | [results/tables/drift_severity.md](results/tables/drift_severity.md) | `results/raw/drift_sweep/` |
| Figure: execution time per window vs window size | [results/figures/timing.pdf](results/figures/timing.pdf) | `results/raw/timing/` |
| Figure: scores over a data stream, global | [results/figures/drift_stream.pdf](results/figures/drift_stream.pdf) | `results/raw/drift_stream/` |
| Figure: scores over a data stream, per label | [results/figures/drift_stream_labels.pdf](results/figures/drift_stream_labels.pdf) | `results/raw/drift_stream/` |

## Citation

```bibtex
@article{shade,
  title  = {SHADE: Lightweight Unsupervised Drift Detection for Constrained Devices},
  author = {TODO},
  year   = {TODO}
}
```

## License

[MIT](LICENSE)
