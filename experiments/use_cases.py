""" The 10 use cases of the paper, the loading of their embeddings, and the default folders.

Every use case has four embedding sets, stored as HDF5 files in data/embeddings/UC<id>/ (see data/README.md):
    train       reference set: the training set of the classifier (detectors are fitted on it)
    test        drift-free set used to estimate the thresholds
    new_unseen  drift-free set from which the monitored windows are sampled
    drift       drifted samples injected in the monitored windows
Each file contains the embeddings E, the original labels Y_original and the labels predicted by the classifier
Y_predicted.
"""
import json
import os

import numpy as np

# Folders, relative to the working directory and found from the location of this file
ROOT = os.path.relpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))  # "." from the repository root
EMBEDDINGS_DIR = os.path.normpath(os.path.join(ROOT, "data", "embeddings"))
RESULTS_DIR = os.path.normpath(os.path.join(ROOT, "results"))
RAW_DIR = os.path.join(RESULTS_DIR, "raw")

# Fields of each use case:
#   dataset, model  names shown in the tables
#   labels          labels of the training classes
#   drift_labels    original labels of the drifted samples (drifted windows are balanced over them)
#   d1, d1_l        DriftLens principal components for the whole window and for each label
#   drift_files     {drift percentage: file}, for the use case with one drift file per drift percentage
#   relabel_drift   original label assigned to every drift sample
#   float32         cast the embeddings to float32
USE_CASES = {
    "01": dict(dataset="AG News", model="BERT", labels=[0, 1, 2], drift_labels=[3], d1=150, d1_l=75),
    "02": dict(dataset="20 Newsgroups", model="BERT", labels=[0, 1, 2, 3, 4], drift_labels=[5], d1=150, d1_l=75),
    "03": dict(dataset="20 Newsgroups (motors)", model="BERT", labels=[0, 1, 2, 3, 4, 5], drift_labels=[6], d1=150, d1_l=75),
    # Bias in Bios: gender-swap drift, the drift set contains the training labels
    "04": dict(dataset="Bias in Bios", model="BERT", labels=[0, 1, 2, 3, 4, 5], drift_labels=[0, 1, 2, 3, 4, 5], d1=150, d1_l=25),
    # MNIST: the drift set contains two new digits (8 and 9), both with the label 8
    "05": dict(dataset="MNIST", model="ViT", labels=[0, 1, 2, 3, 4, 5, 6, 7], drift_labels=[8], d1=150, d1_l=75),
    "06": dict(dataset="Intel Image", model="ViT", labels=[0, 1, 2, 3, 4], drift_labels=[5], d1=150, d1_l=25),
    "07": dict(dataset="STL", model="ViT", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8], drift_labels=[9], d1=150, d1_l=25),
    # STL blur: the drift set contains the training labels; one drift file per drift percentage, and the drifted
    # windows are sampled entirely from the file of their percentage
    "08": dict(dataset="STL blur", model="ViT", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], drift_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               d1=150, d1_l=25, drift_files={0.05: "drift_05.hdf5", 0.10: "drift_10.hdf5"}),
    # FairFace: the drift samples (a new group of people) keep the gender labels, so they are relabeled as class 2
    "09": dict(dataset="FairFace", model="ViT", labels=[0, 1], drift_labels=[2], d1=150, d1_l=25, relabel_drift=2, float32=True),
    "10": dict(dataset="Common Voice", model="Wav2Vec", labels=[0, 1], drift_labels=[2], d1=15, d1_l=25),
}
ALL_USE_CASES = sorted(USE_CASES)


def load_embedding(path):
    """ Loads an embedding file.

    Returns:
        (:obj:`numpy.ndarray`, :obj:`numpy.ndarray`, :obj:`numpy.ndarray`): Embeddings of shape *(n, d)*, original
        labels and predicted labels of shape *(n)*.
    """
    import h5py
    with h5py.File(path, "r") as hf:
        Y_original = hf["Y_original"][()]
        Y_predicted = hf["Y_predicted"][()]
        E = hf["E"][()][0:len(Y_original)]
    return E, Y_original, Y_predicted


class UseCase:
    """ Embeddings and settings of one use case.

    Attributes:
        id, dataset, model      Identifier ("01" ... "10") and names.
        labels, drift_labels    Training labels and original labels of the drifted samples.
        d1, d1_l                DriftLens principal components (whole window, each label).
        E_train, Yp_train       Reference embeddings and their predicted labels.
        E_test, Yp_test         Threshold embeddings and their predicted labels.
        E_new, Yo_new, Yp_new   Drift-free embeddings of the monitored data, original and predicted labels.
        drift                   Drift sets as (E, Y_original, Y_predicted): {"all": set} for a single drift file,
                                {drift percentage: set} for one drift file per drift percentage.
    """
    def __init__(self, uc_id, embeddings_dir=EMBEDDINGS_DIR):
        if uc_id not in USE_CASES:
            raise ValueError(f"Unknown use case '{uc_id}': choose among {', '.join(ALL_USE_CASES)}")
        cfg = USE_CASES[uc_id]
        self.id = uc_id
        self.cfg = cfg
        self.dataset, self.model = cfg["dataset"], cfg["model"]
        self.labels, self.drift_labels = cfg["labels"], cfg["drift_labels"]
        self.d1, self.d1_l = cfg["d1"], cfg["d1_l"]
        self.dir = os.path.join(embeddings_dir, f"UC{uc_id}")
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"{self.dir} not found: download the embeddings with data/download.sh")
        self.E_train, _, self.Yp_train = self.load("train.hdf5")
        self.E_test, _, self.Yp_test = self.load("test.hdf5")
        self.E_new, self.Yo_new, self.Yp_new = self.load("new_unseen.hdf5")
        if "drift_files" in cfg:
            self.drift = {p: self.load(filename) for p, filename in cfg["drift_files"].items()}
        else:
            E_d, Yo_d, Yp_d = self.load("drift.hdf5")
            if "relabel_drift" in cfg:
                Yo_d = np.full(len(Yo_d), cfg["relabel_drift"])
            self.drift = {"all": (E_d, Yo_d, Yp_d)}
        self.n_features = self.E_train.shape[1]
        return

    def load(self, filename):
        E, Y_original, Y_predicted = load_embedding(os.path.join(self.dir, filename))
        if self.cfg.get("float32", False):
            E = E.astype(np.float32)
        return E, Y_original, Y_predicted

    def pool_fraction(self, m_w):
        """ Largest fraction of a label pool (threshold, drift-free or per-percentage drift set) taken by one
        class-balanced window of size m_w. Windows are sampled without repetition, so a size is not feasible when
        the fraction is above 1. """
        pools = [(self.Yp_test, self.labels), (self.Yo_new, self.labels)]
        pools += [(Yo, self.drift_labels) for p, (_, Yo, _) in self.drift.items() if p != "all"]
        fraction = 0.0
        for Y, label_list in pools:
            for label in label_list:
                fraction = max(fraction, np.ceil((1 / len(label_list)) * m_w) / max(np.sum(Y == label), 1))
        return float(fraction)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return


def load_json(path):
    with open(path) as f:
        return json.load(f)
