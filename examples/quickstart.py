""" SHADE quickstart: fit on reference embeddings, estimate a threshold, score windows.

The example uses synthetic embeddings, so it only needs NumPy:
    python examples/quickstart.py
With real data, replace the three arrays E_ref, E_th and the windows with the embeddings of your model.
"""
import numpy as np

from shade import SHADE

rng = np.random.default_rng(0)
d, n_classes, m_w = 256, 4, 500
centres = rng.standard_normal((n_classes + 1, d))  # one centre per class; the last one is an unseen class (the drift)


def embeddings(n, classes):
    """ Synthetic embeddings of n samples of the given classes, with their labels. """
    y = rng.choice(classes, n)
    return centres[y] + rng.standard_normal((n, d)), y


known = np.arange(n_classes)
E_ref, Y_ref = embeddings(20000, known)   # reference set (e.g. the training set of the model)
E_th, Y_th = embeddings(10000, known)     # held-out drift-free set, for the threshold

# 1. Fit: bin edges and reference frequencies on K = 64 random directions, B = 2 bins
shade = SHADE(n_directions=64, n_bins=2, seed=0).fit(E_ref, Y_ref)
print(f"Stored state: {shade.state_nbytes() / 1e3:.1f} kB")

# 2. Threshold: 99th percentile of the scores of drift-free windows
th_scores = [shade.score(E_th[rng.choice(len(E_th), m_w, replace=False)])["global"] for _ in range(1000)]
tau = np.quantile(th_scores, 0.99)
print(f"Threshold: {tau:.3f}")

# 3. Score new windows: without drift, and with 5% and 20% of samples from the unseen class
for drift in (0.0, 0.05, 0.20):
    n_drift = int(drift * m_w)
    E_w, Y_w = embeddings(m_w - n_drift, known)
    E_d, _ = embeddings(n_drift, [n_classes])
    Y_d = rng.choice(known, n_drift)  # the model predicts the drifted samples as one of the known classes
    s = shade.score(np.concatenate([E_w, E_d]), np.concatenate([Y_w, Y_d]))
    per_label = ", ".join(f"{label}: {v:.2f}" for label, v in s["per_label"].items())
    print(f"{drift:4.0%} drift: score {s['global']:6.3f} -> {'DRIFT' if s['global'] > tau else 'no drift'}   (per label: {per_label})")
