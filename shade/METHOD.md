# The SHADE method

SHADE detects distribution drift in a stream of embeddings (the internal representations of a deep learning model)
without labels. It compares each window of new embeddings with a reference set and returns a drift score; the window
is flagged when the score exceeds a threshold estimated on drift-free data. It is designed for constrained devices:
the stored state is a few kilobytes, and a window is summarized by counters that are updated one sample at a time.

The implementation is the class `SHADE` in [detector.py](detector.py).

## 1. In one paragraph

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
2. **Bin edges.** For each direction, the empirical quantiles of $\{r_k^\top x : x \in E_{ref}\}$ at levels $j/B$,
   $j = 1, \dots, B-1$. For $B = 2$ this is the median.
3. **Code** of a sample on direction $k$: $c_k(x) = \#\{j : r_k^\top x > e_{k,j}\} \in \{0, \dots, B-1\}$.
4. **Reference frequencies.** $q_b[k, j]$ is the fraction of $E_{ref}$ with $c_k = j$; it is $\approx 1/B$ by
   construction. The same frequencies are computed for each predicted label, with the same bin edges.

### 2.3 Offline: estimating the threshold

Sample $n_{th}$ drift-free windows of size $m_w$ from $E_{th}$, score them, and set the threshold $\tau$ to the
$(1 - \alpha)$ empirical quantile of their scores. The paper uses $n_{th} = 10000$, $\alpha = 0.01$ and windows with
the same number of samples per label. An empirical threshold accounts for the correlation between directions and for
any systematic difference between the reference set and held-out data (Section 3).

### 2.4 Online: scoring a window

**Counts:** $n_w[k, j] = \#\{x \in W : c_k(x) = j\}$, $q_w = n_w / m_w$.

**Per-direction statistic** (grouped Anderson–Darling), with the cumulative frequencies
$F_w[k, j] = \sum_{i \le j} q_w[k, i]$ and $F_b[k, j] = \sum_{i \le j} q_b[k, i]$ at the $B - 1$ bin edges:

$$A^2_k = m_w \sum_{j=1}^{B-1} q_b[k, j]\, \frac{\big(F_w[k,j] - F_b[k,j]\big)^2}{F_b[k,j]\,\big(1 - F_b[k,j]\big) + \varepsilon}, \qquad \varepsilon = 10^{-3}$$

At each bin edge, the squared gap between the share of the window and the share of the reference below it is divided
by its variance under no drift, so the tails count as much as the centre.

**Window score:** $S(W) = \frac{1}{K} \sum_{k=1}^{K} A^2_k$; drift if $S(W) > \tau$.

**Default ($B = 2$).** With a single edge at the median, $F_b \approx q_b \approx \frac{1}{2}$ and

$$A^2_k = 2\, m_w \big(p_w[k] - \tfrac{1}{2}\big)^2 = \tfrac{1}{2}\, \chi^2_k$$

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

- **Range:** $S \geq 0$; 0 when every cumulative frequency matches the reference. For $B = 2$ the maximum is
  $m_w / 2$ (every sample on the same side of the median on every direction).
- **Without drift, for i.i.d. windows from the reference distribution:** each $A^2_k$ has expectation
  $\sum_{j} q_b[k, j] = (B-1)/B$, because $\mathbb{E}[(F_w - F_b)^2] = F_b(1-F_b)/m_w$ at each bin edge; that is
  $\frac{1}{2}$ for $B = 2$, for any window size.
- **In practice the drift-free scores differ from this value,** for two reasons: the reference is the training set,
  whose embeddings can differ systematically from held-out data (the classifier was fitted on them), which adds a
  term that grows with $m_w$; and class-balanced windows have a lower variance than i.i.d. windows. Thresholds of the
  default configuration measured in the experiments of the paper (99th percentile, mean over 5 runs):

  | UC | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | $m_w = 250$ | 0.45 | 5.15 | 3.98 | 0.31 | 0.34 | 0.48 | 0.36 | 0.29 | 2.92 | 0.98 |
  | $m_w = 1000$ | 0.52 | 17.79 | 13.47 | 0.42 | 0.53 | 0.80 | 0.52 | 0.41 | 7.99 | 2.18 |

  This is why the threshold is estimated empirically rather than taken from the χ² distribution.
- **The threshold set must be representative of the drift-free monitored data.** When the two differ, the false
  alarms rise (in UC10 the accuracy on drift-free windows drops to 0.35 at $m_w = 2000$) or the threshold is too
  high and small drifts are missed.

## 4. Why this design

| Choice | Justification |
|---|---|
| Random projections | Sharp Cramér–Wold theorem (Cuesta-Albertos, Fraiman & Ransford 2007): if two distributions (with moments satisfying Carleman's condition, e.g. bounded embeddings) differ, their projections differ on almost every direction. $K$ reduces the variance of the score. |
| Bin edges at the reference quantiles | Equiprobable bins are the standard choice for binned goodness-of-fit tests: every bin and direction is equally informative, whatever the scale of the projection. |
| $B = 2$ by default | The median bit tests whether a projected median moved. Changes of spread or shape that keep the medians are invisible to it; $B > 2$ sees them, at a higher cost (Section 5). |
| Anderson–Darling instead of χ² | A shift moves every cumulative frequency in the same direction; Anderson–Darling accumulates these deviations, while χ² compares the bins independently and ignores their order. The two are identical for $B = 2$. |
| Mean over directions, not maximum | For Gaussian $r$, $r^\top \delta \sim \mathcal{N}(0, \lVert\delta\rVert^2)$: a change is spread over all directions (a dense alternative), where sum-type tests (Chen & Qin 2010) are more powerful than max-type tests (Cai, Liu & Xia 2014). |
| No whitening | Inverting an estimated covariance hurts power in high dimension (Bai & Saranadasa 1996). |
| Empirical threshold | Accounts for the correlation between directions, unlike a Bonferroni correction over the directions (Rabanser et al. 2019), and for the offset of the training reference (Section 3). |

**Relation to SimHash (Charikar 2002).** For $B = 2$, the code is the sign of a random projection offset by the
reference median, i.e. a median-centred SimHash bit; for $B > 2$ it is a multi-bit quantized code. The detection
principle is not locality-sensitive hashing (no fingerprints are compared): it is a sliced two-sample test on the code
frequencies.

## 5. Configurations

The method has two parameters, $K$ and $B$. All configurations use the same code and the same statistic.

| | **Default** ($K = 64$, $B = 2$) | $K = 64$, $B = 8$ | $K = 128$, $B = 2$ | $K = 128$, $B = 8$ |
|---|---|---|---|---|
| Code per direction | 1 bit (median) | 3 bits (octiles) | 1 bit | 3 bits |
| Detects | shifts of the projected medians | changes at the octiles (location, spread, shape) | as default | as $B = 8$ |
| AUC@5% | 0.841 | 0.845 | 0.847 | 0.851 |
| H_DD | 0.683 | 0.691 | 0.722 | 0.722 |
| Acc@0% (target 0.99) | 0.918 | 0.935 | 0.887 | 0.896 |
| Std of AUC@5% across seeds | 0.027 | 0.024 | 0.017 | 0.016 |
| ms per window | 2.48 | 2.98 | 4.27 | 5.39 |
| Stored state | 7.1 kB | 29.5 kB | 14.2 kB | 59.1 kB |

Means over the 10 use cases and 4 window sizes of the paper, 5 runs each
([results/tables/variants.md](../results/tables/variants.md)).

**Why $K = 64$, $B = 2$ is the default:** the four configurations are within 0.01 AUC of each other, and the default
is the cheapest. $K = 128$ lowers the dependence on the random seed and raises H_DD, at 1.7× the time and with more
false alarms. $B = 8$ helps where drift changes the spread or the tails (UC06: AUC@5% 0.89 → 0.95 at $m_w = 2000$) and
lowers the false alarms, but hurts elsewhere (UC10: 0.91 → 0.82) and costs 4× the memory.

## 6. Cost

**Per sample:** $d \cdot K$ multiply-adds for the projection (49k for $d = 768$, $K = 64$), $K \log_2 B$ comparisons
(one per direction for $B = 2$), $K$ counter increments. **Per window:** $O(K \cdot B)$ for the score; no PCA, no
covariance matrix, no matrix square root.

**Stored state** (`SHADE.state_nbytes()`; $R$ is generated from the seed): $K (B - 1)$ bin edges, plus $K \cdot B$
frequencies for the whole reference and for each label. For the default configuration, against DriftLens (mean vectors,
covariance matrices and PCA models):

| UC | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SHADE (kB) | 4.6 | 6.7 | 7.7 | 7.7 | 9.5 | 6.4 | 10.5 | 11.5 | 3.3 | 3.3 | 7.1 |
| DriftLens (kB) | 2647 | 3672 | 4184 | 2099 | 2875 | 1069 | 1409 | 1494 | 814 | 221 | 2048 |

If $R$ cannot be generated on the device, it adds $d \cdot K$ values (197 kB in float32 for $d = 768$, $K = 64$).
Storing the frequencies as float32 or as integer counts would halve the figures of SHADE.

**Time per window** (UC01, $d = 768$, single thread, global and per-label scores, mean over 5 runs;
[results/figures/timing.pdf](../results/figures/timing.pdf)):

| $m_w$ | 250 | 1000 | 2000 | 5000 |
|---|---|---|---|---|
| DriftLens | 12.3 ms | 22.0 ms | 35.1 ms | 72.1 ms |
| SHADE | 1.0 ms | 3.7 ms | 7.1 ms | 17.7 ms |
| Speed-up | 12.0× | 6.0× | 4.9× | 4.1× |

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
- Z. Bai, H. Saranadasa. *Effect of high dimension: by an example of a two sample problem.* Statistica Sinica 6, 1996.
- S. X. Chen, Y.-L. Qin. *A two-sample test for high-dimensional data with applications to gene-set testing.* Ann. Statist., 2010.
- T. T. Cai, W. Liu, Y. Xia. *Two-sample test of high dimensional means under dependence.* JRSS-B, 2014.
- S. Rabanser, S. Günnemann, Z. Lipton. *Failing Loudly: An empirical study of methods for detecting dataset shift.* NeurIPS 2019.
- DriftLens: *Unsupervised Concept Drift Detection From Deep Learning Representations in Real-Time.* IEEE Transactions on Knowledge and Data Engineering 37(10), 2025.
