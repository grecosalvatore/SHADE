# The SHADE method

SHADE detects distribution drift in a stream of embeddings (the internal representations of a deep learning model)
without labels. It compares each window of new embeddings with a reference set and returns a drift score; the window
is flagged when the score exceeds a threshold estimated on drift-free data. It is designed for constrained devices:
the stored state is a few kilobytes, and a window is summarized by counters that are updated one sample at a time.

The implementation is the class `SHADE` in [detector.py](detector.py).

## 1. Framework Overview

Every embedding is projected onto $K$ random directions. On each direction, the reference projections are split into
$B$ equiprobable bins at their quantiles, and each sample gets a **code**: the bin its projection falls in. With the
default $B = 2$, the code is one bit, above or below the reference median (a median-centred SimHash bit). A window is
summarized by $K \times B$ counters. On each direction, the window is compared with the reference by a **grouped
Anderson–Darling** statistic on the cumulative bin frequencies; the statistics are averaged over the directions and
compared with a threshold estimated on drift-free windows. With $B = 2$, the statistic is half of Pearson's χ² on the
bit counts, so the default SHADE is a χ² test of the median split along each random direction.

## 2. Method

### 2.1 Notation

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^d$ | an embedding ($d = 768$ in the use cases of the paper) |
| $E_{ref}$ | reference embeddings (the training set of the monitored classifier), with predicted labels |
| $E_{th}$ | drift-free embeddings for the threshold (a held-out set) |
| $W$ | a window of $m_w$ embeddings, with predicted labels |
| $r_1, \dots, r_K$ | random directions, $r_k \sim \mathcal{N}(0, I_d)$, generated from a seed |
| $B$ | bins per direction |
| $e_{k,1} < \dots < e_{k,B-1}$ | reference quantiles of direction $k$ at levels $j/B$ (the bin edges) |

### 2.2 Offline: building the reference

1. **Directions.** Generate $R = [r_1, \dots, r_K] \in \mathbb{R}^{d \times K}$ from the seed
   (`np.random.default_rng(seed).standard_normal((d, K))`). $R$ does not need to be stored.
2. **Bin edges.** For each direction, the empirical quantiles of $\lbrace r_k^\top x : x \in E_{ref} \rbrace$ at levels $j/B$,
   $j = 1, \dots, B-1$. For $B = 2$ this is the median.
3. **Code** of a sample on direction $k$: $c_k(x) = \lvert \lbrace j : r_k^\top x > e_{k,j} \rbrace \rvert \in \lbrace 0, \dots, B-1 \rbrace$.
4. **Reference frequencies.** $q_b[k, j]$ is the fraction of $E_{ref}$ with $c_k = j$; it is $\approx 1/B$ by
   construction. The same frequencies are computed for each predicted label, with the same bin edges.

### 2.3 Offline: estimating the threshold

Sample $n_{th}$ drift-free windows of size $m_w$ from $E_{th}$, score them, and set the threshold $\tau$ to the
$(1 - \alpha)$ empirical quantile of their scores. The paper uses $n_{th} = 10000$, $\alpha = 0.01$ and windows with
the same number of samples per label. An empirical threshold accounts for the correlation between directions and for
any systematic difference between the reference set and held-out data (Section 3).

### 2.4 Online: scoring a window

**Counts:** $n_w[k, j] = \lvert \lbrace x \in W : c_k(x) = j \rbrace \rvert$, $q_w = n_w / m_w$.

**Per-direction statistic** (grouped Anderson–Darling), with the cumulative frequencies
$F_w[k, j] = \sum_{i \le j} q_w[k, i]$ and $F_b[k, j] = \sum_{i \le j} q_b[k, i]$ at the $B - 1$ bin edges:

$$A^2_k = m_w \sum_{j=1}^{B-1} q_b[k, j] \frac{\big(F_w[k,j] - F_b[k,j]\big)^2}{F_b[k,j] \big(1 - F_b[k,j]\big) + \varepsilon}, \qquad \varepsilon = 10^{-3}$$

At each bin edge, the squared gap between the share of the window and the share of the reference below it is divided
by its variance under no drift, so the tails count as much as the centre.

**Window score:** $S(W) = \frac{1}{K} \sum_{k=1}^{K} A^2_k$; drift if $S(W) > \tau$.

**Default ($B = 2$).** With a single edge at the median, $F_b \approx q_b \approx \frac{1}{2}$ and

$$A^2_k = 2 m_w \big(p_w[k] - \tfrac{1}{2}\big)^2 = \tfrac{1}{2} \chi^2_k$$

where $p_w[k]$ is the fraction of the window above the median and $\chi^2_k$ is Pearson's χ² on the two bit counts
(up to $\varepsilon$).

### 2.5 Per-label scores

The same counts restricted to the window samples predicted as label $\ell$, compared with the reference frequencies
of $\ell$ (same bin edges), with $m_w$ replaced by the number of such samples; NaN if the label is absent from the
window. With a threshold per label, estimated like the global one, they show which predicted classes the drifted
samples fall in.

### 2.6 Streaming form

The window never needs to be stored. The state is $K \cdot B$ counters, plus the same per label:

```
on each new sample x with predicted label y:
    z = Rᵀ x                                  # K dot products
    for k in 1..K:
        c = binary_search(edges[k], z[k])     # log2(B) comparisons (one comparison for B = 2)
        counts[k, c] += 1
        counts_label[y][k, c] += 1
when m_w samples have arrived:
    S = mean_k A²_k(counts, ref_freq)         # O(K · B)
    raise an alarm if S > τ ; optionally score each label
    reset the counters
```

The Python implementation scores a whole window at once (vectorized, with $K(B-1)$ comparisons per sample), which
gives the same scores.

## 3. Interpreting the score

The score is non-negative, and 0 when every cumulative frequency of the window matches the reference; for $B = 2$ its
maximum is $m_w / 2$. Its distribution on drift-free windows depends on the data (the reference is the training set of
the classifier, and the windows are class-balanced), so the threshold is estimated empirically on drift-free windows
rather than taken from the χ² distribution.

## 4. Design choices

| Choice | Justification |
|---|---|
| Random projections | Sharp Cramér–Wold theorem (Cuesta-Albertos, Fraiman & Ransford 2007): if two distributions (with moments satisfying Carleman's condition, e.g. bounded embeddings) differ, their projections differ on almost every direction. |
| Bin edges at the reference quantiles | Equiprobable bins are the standard choice for binned goodness-of-fit tests: every bin and direction is equally informative, whatever the scale of the projection. |
| Anderson–Darling instead of χ² | A shift moves every cumulative frequency in the same direction; Anderson–Darling accumulates these deviations, while χ² compares the bins independently and ignores their order. The two are identical for $B = 2$. |

## 5. Parameters

| Parameter | Meaning | Default |
|---|---|---|
| $K$ (`n_directions`) | number of random directions | 64 |
| $B$ (`n_bins`) | equiprobable bins per direction: $B = 2$ splits at the median (1 bit), $B = 8$ at the octiles (3 bits) | 2 |
| `seed` | seed of the random directions | 0 |
| `eps` | constant $\varepsilon$ in the denominator of the statistic | $10^{-3}$ |

The paper evaluates $K \in \lbrace 64, 128 \rbrace$ and $B \in \lbrace 2, 8 \rbrace$:
[results/tables/variants.md](../results/tables/variants.md) ([.tex](../results/tables/variants.tex)).

## 6. Cost

**Per sample:** $d \cdot K$ multiply-adds for the projection, $K \log_2 B$ comparisons and $K$ counter increments.
**Per window:** $O(K \cdot B)$ for the score.

**Stored state** (`SHADE.state_nbytes()`): $K (B - 1)$ bin edges, plus $K \cdot B$ reference frequencies for the whole
reference and for each label. The directions $R$ are generated from the seed; if they cannot be generated on the
device, they add $d \cdot K$ values.

Measured stored state and time per window: [results/tables/variants.md](../results/tables/variants.md) (columns
Stored and ms/window) and [results/figures/timing.pdf](../results/figures/timing.pdf) (time per window vs window size).

## 7. Usage

```python
import numpy as np
from shade import SHADE

shade = SHADE(n_directions=64, n_bins=2, seed=0)        # default: K = 64, B = 2
shade.fit(E_ref, Y_ref)                                 # bin edges and reference frequencies
scores = shade.score(E_w, Y_w)                          # {"global": S, "per_label": {label: S_label}}
drift = scores["global"] > tau                          # tau: 99th percentile of the scores of drift-free windows
```

`eps` only matters for the per-label references, where the cumulative frequencies of a label can approach 0 or 1.
A complete example, with the estimation of the threshold, is in
[examples/quickstart.py](../examples/quickstart.py).

## References

- M. Charikar. *Similarity estimation techniques from rounding algorithms.* STOC 2002.
- T. W. Anderson, D. A. Darling. *Asymptotic theory of certain "goodness of fit" criteria based on stochastic processes.* Ann. Math. Stat., 1952.
- V. Choulakian, R. A. Lockhart, M. A. Stephens. *Cramér–von Mises statistics for discrete distributions.* Canadian Journal of Statistics 22(1), 1994.
- J. A. Cuesta-Albertos, R. Fraiman, T. Ransford. *A sharp form of the Cramér–Wold theorem.* J. Theoret. Probab. 20(2), 2007.
