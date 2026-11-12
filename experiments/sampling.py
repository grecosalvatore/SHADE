""" Class-balanced window sampling and drift injection.

A window of size m_w with drift percentage p contains (1 - p) m_w drift-free samples, with the same number of samples
per label, and p m_w drifted samples, with the same number of samples per drift label. Samples are drawn without
repetition within a window, and the window is shuffled.

Seeds of run r:
    detection   1000 r + 0 threshold windows, + 1 drift-free windows, + 2 drifted windows
    drift sweep 1000 r + 100 drift-free windows, + 101 drifted windows
    stream      1000 r + 200 + i for the i-th drift pattern
"""
import numpy as np

DRIFT_PERCENTAGES = [0.05, 0.10]


class WindowSampler:
    """ Samples class-balanced windows, optionally with a percentage of drifted samples.

    Args:
        labels              (:obj:`list`): Labels of the drift-free samples.
        drift_labels        (:obj:`list`): Labels of the drifted samples.
        E, Y_predicted      (:obj:`numpy.ndarray`): Drift-free embeddings and their predicted labels.
        Y_balance           (:obj:`numpy.ndarray`): Labels used to balance the drift-free samples.
        E_drift, Y_predicted_drift, Y_balance_drift (:obj:`numpy.ndarray`, optional): The same for the drifted samples.
        seed                (:obj:`int`): Seed of the sampler.
    """
    def __init__(self, labels, drift_labels, E, Y_predicted, Y_balance, E_drift=None, Y_predicted_drift=None,
                 Y_balance_drift=None, seed=0):
        self.rng = np.random.default_rng(seed)
        self.labels = labels
        self.drift_labels = drift_labels
        self.E, self.Y_predicted = E, Y_predicted
        self.E_drift, self.Y_predicted_drift = E_drift, Y_predicted_drift
        self.idx = {l: np.flatnonzero(Y_balance == l) for l in labels}
        if E_drift is not None:
            self.drift_idx = {l: np.flatnonzero(Y_balance_drift == l) for l in drift_labels}
        return

    def _draw(self, idx_dict, label_list, n):
        """ Indices of n samples, n // n_labels per label; the remainder goes to labels drawn at random. """
        counts = {l: n // len(label_list) for l in label_list}
        for l in self.rng.choice(label_list, n % len(label_list), replace=True):
            counts[l] += 1
        return np.concatenate([self.rng.choice(idx_dict[l], counts[l], replace=False) for l in label_list])

    def window(self, window_size, drift_percentage=0.0):
        """ Embeddings and predicted labels of a new window with the given percentage of drifted samples. """
        m_drift = int(round(window_size * drift_percentage))
        idx = self._draw(self.idx, self.labels, window_size - m_drift)
        E_w, Y_w = self.E[idx], self.Y_predicted[idx]
        if m_drift > 0:
            idx_d = self._draw(self.drift_idx, self.drift_labels, m_drift)
            E_w = np.concatenate([E_w, self.E_drift[idx_d]])
            Y_w = np.concatenate([Y_w, self.Y_predicted_drift[idx_d]])
        p = self.rng.permutation(len(E_w))
        return E_w[p], Y_w[p]


def threshold_sampler(uc, run):
    """ Sampler of the drift-free threshold windows of a run, balanced by predicted label. """
    return WindowSampler(uc.labels, uc.drift_labels, uc.E_test, uc.Yp_test, uc.Yp_test, seed=1000 * run)


def detection_samplers(uc, run):
    """ Samplers of the monitored windows of a run: {drift percentage: (sampler, percentage injected by the sampler)},
    0.0 for the drift-free windows. With one drift file per drift percentage, the drifted windows are sampled entirely
    from that file, so the sampler injects nothing. """
    base = 1000 * run
    out = {0.0: (WindowSampler(uc.labels, uc.drift_labels, uc.E_new, uc.Yp_new, uc.Yo_new, seed=base + 1), 0.0)}
    if "all" in uc.drift:
        E_d, Yo_d, Yp_d = uc.drift["all"]
        sampler = WindowSampler(uc.labels, uc.drift_labels, uc.E_new, uc.Yp_new, uc.Yo_new, E_d, Yp_d, Yo_d, seed=base + 2)
        out.update({p: (sampler, p) for p in DRIFT_PERCENTAGES})
    else:
        for i, p in enumerate(DRIFT_PERCENTAGES):
            E_d, Yo_d, Yp_d = uc.drift[p]
            out[p] = (WindowSampler(uc.labels, uc.drift_labels, E_d, Yp_d, Yo_d, seed=base + 2 + i), 0.0)
    return out


def mixing_sampler(uc, seed):
    """ Sampler that mixes drift-free samples with any percentage of drifted samples. With one drift file per drift
    percentage, the drifted samples come from the file of the lowest percentage. """
    E_d, Yo_d, Yp_d = uc.drift["all"] if "all" in uc.drift else uc.drift[min(uc.drift)]
    return WindowSampler(uc.labels, uc.drift_labels, uc.E_new, uc.Yp_new, uc.Yo_new, E_d, Yp_d, Yo_d, seed=seed)


def sweep_samplers(uc, run, levels):
    """ Samplers of the drift sweep of a run: {drift percentage: (sampler, percentage injected by the sampler)}. """
    base = 1000 * run + 100
    out = {0.0: (WindowSampler(uc.labels, uc.drift_labels, uc.E_new, uc.Yp_new, uc.Yo_new, seed=base), 0.0)}
    sampler = mixing_sampler(uc, base + 1)
    out.update({p: (sampler, p) for p in levels if p > 0})
    return out


def stream_sampler(uc, run, pattern_index):
    """ Sampler of the stream of a drift pattern of a run. """
    return mixing_sampler(uc, 1000 * run + 200 + pattern_index)
