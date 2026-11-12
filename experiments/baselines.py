""" DriftLens baseline: adapters that give the DriftLens library the interface of the SHADE detector.

DriftLens reduces the embeddings with PCA, models the reference and each window as multivariate Gaussians and scores
a window with the Frechet distance between the two, for the whole window and for each predicted label.

Interface of every detector used by the experiments:
    fit(E, Y, labels)   builds the reference
    score(E_w, Y_w)     {"global": score of the window, "per_label": {label: score}}
    state_nbytes()      size in bytes of the state stored after fit
"""
import os
import sys

import numpy as np

# Location of the DriftLens source fetched by experiments/scripts/get_driftlens.sh, used when the library is not
# installed in the environment
_DRIFTLENS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "third_party", "drift-lens")

try:
    from driftlens.driftlens import DriftLens
    from driftlens.distribution_distances import frechet_drift_distance as fdd
except ImportError:
    if not os.path.isdir(os.path.join(_DRIFTLENS_SRC, "driftlens")):
        raise ImportError("DriftLens not found: run 'bash experiments/scripts/get_driftlens.sh'") from None
    sys.path.insert(0, os.path.normpath(_DRIFTLENS_SRC))
    from driftlens.driftlens import DriftLens
    from driftlens.distribution_distances import frechet_drift_distance as fdd


class DriftLensDetector:
    """ DriftLens with the Frechet distance, global and per-label scores.

    Args:
        d1      (:obj:`int`): Principal components for the whole window.
        d1_l    (:obj:`int`): Principal components for each label.
    """
    def __init__(self, d1, d1_l):
        self.dl = DriftLens()
        self.d1, self.d1_l = d1, d1_l
        self.baseline = None
        self.labels = None
        return

    def fit(self, E, Y, labels):
        self.labels = labels
        self.baseline = self.dl.estimate_baseline(E=E, Y=Y, label_list=labels, batch_n_pc=self.d1, per_label_n_pc=self.d1_l)
        return self

    def score(self, E_w, Y_w):
        """ Global and per-label distances of the library. It raises a ValueError when a label has too few samples
        in the window for its distance to be computed. """
        d = self.dl.compute_window_distribution_distances(E_w, Y_w)
        return {"global": float(np.real(d["per-batch"])),
                "per_label": {l: float(np.real(d["per-label"][str(l)])) for l in self.labels}}

    def score_global(self, E_w):
        """ Global distance only, computed as in the library: PCA of the whole window, then Frechet distance. """
        b = self.baseline
        E_r = b.get_batch_PCA_model().transform(E_w)
        d = fdd.frechet_distance(b.get_batch_mean_vector(), fdd.get_mean(E_r), b.get_batch_covariance_matrix(), fdd.get_covariance(E_r))
        return float(np.real(d))

    def state_nbytes(self):
        """ Mean vectors and covariance matrices of the reference, plus the PCA models needed to reduce the windows
        (components and means), for the whole window and for each label. """
        b = self.baseline
        distributions = (b.get_batch_mean_vector().nbytes + b.get_batch_covariance_matrix().nbytes
                         + sum(m.nbytes for m in b.get_per_label_mean_vectors().values())
                         + sum(c.nbytes for c in b.get_per_label_covariance_matrices().values()))
        pcas = [b.get_batch_PCA_model()] + [b.get_PCA_model_by_label(l) for l in self.labels]
        return int(distributions + sum(p.components_.nbytes + p.mean_.nbytes for p in pcas))


class GlobalDriftLens:
    """ View of a fitted DriftLensDetector that computes the global score only. With a high percentage of drifted
    samples a window can miss some predicted labels, and the per-label distances of the library cannot be computed;
    the global score does not depend on them. """
    def __init__(self, detector):
        self.detector = detector
        return

    def score(self, E_w, Y_w=None):
        return {"global": self.detector.score_global(E_w), "per_label": {}}


class RobustDriftLens:
    """ View of a fitted DriftLensDetector that returns the global score alone, with NaN per-label scores, for the
    windows where the per-label distances of the library cannot be computed. """
    def __init__(self, detector):
        self.detector = detector
        return

    def score(self, E_w, Y_w):
        try:
            return self.detector.score(E_w, Y_w)
        except ValueError:
            return {"global": self.detector.score_global(E_w), "per_label": {l: np.nan for l in self.detector.labels}}
