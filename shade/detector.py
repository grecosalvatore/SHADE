""" SHADE drift detector.

Embeddings are projected on K random directions. On each direction the reference projections are split into B
equiprobable bins, and a window is summarized by its bin counts. The drift score is the grouped Anderson-Darling
statistic between the cumulative bin frequencies of the window and of the reference, averaged over the directions.
See METHOD.md in this folder for the full description.
"""
import numpy as np


class SHADE:
    """ SHADE drift detector.

    Args:
        n_directions (:obj:`int`): Number of random directions K.
        n_bins       (:obj:`int`): Number of equiprobable bins per direction B (B = 2 splits each direction at the
                                   reference median).
        seed         (:obj:`int`): Seed of the random directions.
        eps          (:obj:`float`): Constant added to the denominator of the statistic.

    Attributes (available after :meth:`fit`):
        directions  (:obj:`numpy.ndarray`): Random directions of shape *(n_features, K)*. They are generated from the
                                            seed, so they are not part of the stored state.
        edges       (:obj:`numpy.ndarray`): Reference quantiles (bin edges) of shape *(K, B - 1)*.
        ref_freq    (:obj:`numpy.ndarray`): Reference bin frequencies of shape *(K, B)*.
        ref_freq_per_label (:obj:`dict`): Reference bin frequencies of each label, {label: array of shape *(K, B)*}.
        labels      (:obj:`list`): Labels with a per-label reference.
    """
    def __init__(self, n_directions=64, n_bins=2, seed=0, eps=1e-3):
        if n_directions < 1:
            raise ValueError("n_directions must be at least 1")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self.n_directions = n_directions
        self.n_bins = n_bins
        self.seed = seed
        self.eps = eps
        self.directions = None
        self.edges = None
        self.ref_freq = None
        self.ref_freq_per_label = {}
        self.labels = []
        self._offsets = n_bins * np.arange(n_directions)
        return

    def fit(self, E, Y=None, labels=None):
        """ Estimates the bin edges and the reference bin frequencies.

        Args:
            E       (:obj:`numpy.ndarray`): Reference embeddings of shape *(n_samples, n_features)*.
            Y       (:obj:`numpy.ndarray`, optional): Predicted labels of the reference embeddings, of shape
                                                      *(n_samples)*. Without them only the global score is available.
            labels  (:obj:`list`, optional): Labels to build a per-label reference for (default: the labels in Y).

        Returns:
            :obj:`SHADE`: The fitted detector.
        """
        E = np.asarray(E)
        K, B = self.n_directions, self.n_bins
        self.directions = np.random.default_rng(self.seed).standard_normal((E.shape[1], K)).astype(E.dtype, copy=False)
        self.edges = np.quantile(E @ self.directions, np.arange(1, B) / B, axis=0).T.astype(E.dtype)
        codes = self._codes(E)
        self.ref_freq = self._counts(codes) / len(E)
        self.labels, self.ref_freq_per_label = [], {}
        if Y is not None:
            Y = np.asarray(Y)
            self.labels = list(labels) if labels is not None else np.unique(Y).tolist()
            for label in self.labels:
                mask = Y == label
                self.ref_freq_per_label[label] = self._counts(codes[mask]) / max(np.sum(mask), 1)
        return self

    def score(self, E_w, Y_w=None):
        """ Drift score of a window.

        Args:
            E_w (:obj:`numpy.ndarray`): Window embeddings of shape *(m_w, n_features)*.
            Y_w (:obj:`numpy.ndarray`, optional): Predicted labels of the window, of shape *(m_w)*.

        Returns:
            :obj:`dict`: {"global": score of the window, "per_label": {label: score of the window samples predicted
            with that label}}. "per_label" is empty when Y_w is not given or the detector was fitted without labels;
            the score of a label that is absent from the window is NaN.
        """
        if self.edges is None:
            raise RuntimeError("fit must be called before score")
        codes = self._codes(np.asarray(E_w))
        per_label = {}
        if Y_w is not None:
            Y_w = np.asarray(Y_w)
            per_label = {label: self._statistic(self._counts(codes[Y_w == label]), self.ref_freq_per_label[label])
                         for label in self.labels}
        return {"global": self._statistic(self._counts(codes), self.ref_freq), "per_label": per_label}

    def state_nbytes(self):
        """ Size in bytes of the stored state: bin edges and reference frequencies (global and per label). The random
        directions are excluded, since they are generated from the seed. """
        return int(self.edges.nbytes + self.ref_freq.nbytes + sum(f.nbytes for f in self.ref_freq_per_label.values()))

    def _codes(self, E):
        """ Bin index in [0, B) of every embedding on every direction, of shape *(m, K)*. """
        return ((E @ self.directions)[:, :, None] > self.edges[None]).sum(-1)

    def _counts(self, codes):
        """ Bin counts per direction, of shape *(K, B)*. """
        K, B = self.n_directions, self.n_bins
        return np.bincount((codes + self._offsets).ravel(), minlength=K * B).reshape(K, B)

    def _statistic(self, counts, q_b):
        """ Grouped Anderson-Darling statistic between the window counts and the reference frequencies q_b, averaged
        over the directions. NaN for an empty window. """
        m = counts[0].sum()
        if m == 0:
            return np.nan
        F_w, F_b = np.cumsum(counts / m, 1)[:, :-1], np.cumsum(q_b, 1)[:, :-1]
        return m * np.mean(np.sum((F_w - F_b) ** 2 * q_b[:, :-1] / (F_b * (1 - F_b) + self.eps), axis=1))
