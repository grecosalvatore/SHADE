""" Evaluation metrics.

Detection: a window is flagged as drifted when its score exceeds the threshold, the (1 - T_ALPHA) quantile of the scores
of the drift-free threshold windows.
    Acc@Q   accuracy on the windows with drift percentage Q (share of windows not flagged for Q = 0, flagged otherwise)
    H_DD    harmonic mean of Acc@0 and of the mean accuracy over the drift percentages Q > 0
    AUC@Q   area under the ROC curve of the scores, drift-free windows vs windows with drift percentage Q
Drift severity:
    rho     Spearman correlation between the drift percentage of a window and its score
    onset   smallest drift percentage at which the mean score of the windows exceeds the threshold
"""
import numpy as np

T_ALPHA = 0.01


def threshold(th_scores, t_alpha=T_ALPHA):
    """ Threshold: the (1 - t_alpha) quantile of the scores of the threshold windows (NaN scores are ignored). """
    return float(np.nanquantile(th_scores, 1 - t_alpha))


def hdd(accuracy):
    """ H_DD and false-alarm rate from the accuracies {drift percentage (str): accuracy}. """
    a0 = accuracy["0.0"]
    ad = float(np.mean([v for k, v in accuracy.items() if k != "0.0"]))
    return {"far": 1 - a0, "hdd": 0.0 if a0 + ad == 0 else 2 * a0 * ad / (a0 + ad)}


def evaluate(th_scores, online_scores, t_alpha=T_ALPHA):
    """ Threshold, Acc@Q, AUC@Q, false-alarm rate and H_DD of a detector.

    Args:
        th_scores       (:obj:`numpy.ndarray`): Scores of the threshold windows.
        online_scores   (:obj:`dict`): Scores of the monitored windows {drift percentage: scores}, 0.0 for no drift.

    Returns:
        :obj:`dict`: {"threshold", "accuracy": {Q: Acc@Q}, "auc": {Q: AUC@Q}, "far", "hdd"}.
    """
    from sklearn.metrics import roc_auc_score
    tau = threshold(th_scores, t_alpha)
    out = {"threshold": tau, "accuracy": {}, "auc": {}}
    for p, s in online_scores.items():
        out["accuracy"][str(p)] = float(np.mean(s <= tau) if p == 0 else np.mean(s > tau))
        if p > 0:
            out["auc"][str(p)] = float(roc_auc_score(np.r_[np.zeros(len(online_scores[0.0])), np.ones(len(s))],
                                                     np.r_[online_scores[0.0], s]))
    out.update(hdd(out["accuracy"]))
    return out


def spearman_rho(level_scores, max_level=1.0):
    """ Spearman correlation between drift percentage and score over all the windows with drift percentage up to
    max_level. level_scores: {drift percentage: scores of its windows}. """
    from scipy.stats import spearmanr
    x = np.concatenate([np.full(len(s), p) for p, s in level_scores.items() if p <= max_level + 1e-9])
    y = np.concatenate([s for p, s in level_scores.items() if p <= max_level + 1e-9])
    ok = ~np.isnan(y)
    return float(spearmanr(x[ok], y[ok]).statistic)


def onset(levels, mean_ratio):
    """ Smallest drift percentage p > 0 whose mean score / threshold exceeds 1, None if there is none.
    mean_ratio: {str(p): mean score / threshold}. """
    for p in levels:
        if p > 0 and mean_ratio[str(p)] > 1:
            return p
    return None


def mean_std(values):
    """ [mean, standard deviation] of the values, ignoring NaN. """
    v = np.asarray(values, dtype=float)
    return [float(np.nanmean(v)), float(np.nanstd(v))]
