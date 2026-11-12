# Embeddings

The experiments run on pre-computed embeddings of the 10 use cases of the paper. They are distributed as one archive,
`shade_embeddings.zip` (about 2.7 GB), hosted on
[Google Drive](https://drive.google.com/file/d/1mP2wBbt5I4ryFdWgkolC9msms_xqEJWv).

```bash
bash data/download.sh                              # download with gdown, unzip into data/embeddings, verify
bash data/download.sh path/to/shade_embeddings.zip # or: unzip and verify an archive downloaded by hand
```

`data/embeddings/` is not tracked by git.

## Content

```
data/embeddings/
├── SHA256SUMS                      checksums of every file
├── UC01/ … UC10/
│   ├── train.hdf5                  reference set: the training set of the classifier
│   ├── test.hdf5                   drift-free set used to estimate the thresholds
│   ├── new_unseen.hdf5             drift-free set from which the monitored windows are sampled
│   └── drift.hdf5                  drifted samples injected in the monitored windows
└── UC08/drift_05.hdf5, drift_10.hdf5   UC08 has one drift file per drift percentage instead of drift.hdf5
```

Every file is an HDF5 file with the datasets read by the code:

| Dataset | Shape | Content |
|---|---|---|
| `E` | (n, 768) | embeddings: the last hidden representation of the classifier for each sample |
| `Y_original` | (n) | ground-truth label ids |
| `Y_predicted` | (n) | label ids predicted by the classifier |

Some files contain additional datasets that the code does not read: label names, the input texts (`X`, UC01–UC03),
sample ids (`X`, UC09) and speaker accents (UC10).

## Use cases

Each use case is a classifier fine-tuned on a subset of a dataset (the training classes) and a drift: data that the
classifier did not see during training. The embeddings and the predicted labels come from that classifier.

| UC | Dataset | Model | Training classes | Drift | Samples: train / test / new unseen / drift | Size |
|---|---|---|---|---|---|---|
| 01 | AG News | BERT | 3 topics (World, Sports, Business) | a new topic (Sci/Tech) | 59,480 / 5,700 / 30,520 / 31,900 | 486 MB |
| 02 | 20 Newsgroups | BERT | 5 topics (Technology, Sale-Ads, Politics, Religion, Science) | a new topic (Recreation) | 5,080 / 3,387 / 5,560 / 3,655 | 85 MB |
| 03 | 20 Newsgroups (motors) | BERT | 6 topics (the 5 of UC02 and Recreation.sports) | a new sub-topic (Recreation.autos) | 5,744 / 3,829 / 6,304 / 1,805 | 85 MB |
| 04 | Bias in Bios | BERT | 6 occupations, each with biographies of one gender | biographies of the opposite gender (same classes) | 39,270 / 15,108 / 94,184 / 106,242 | 901 MB |
| 05 | MNIST | ViT | 8 digits (0 to 7) | two new digits (8 and 9) | 35,000 / 8,000 / 12,000 / 13,000 | 195 MB |
| 06 | Intel Image | ViT | 5 scenes (Forest, Building, Glacier, Street, Mountain) | a new scene (Sea) | 6,000 / 4,000 / 4,259 / 2,780 | 49 MB |
| 07 | STL | ViT | 9 object classes (all but Truck) | a new class (Truck) | 5,850 / 2,925 / 2,925 / 1,300 | 37 MB |
| 08 | STL blur | ViT | 10 object classes | images with Gaussian blur (same classes) | 6,500 / 3,250 / 3,250 / 3,250 + 3,250 | 55 MB |
| 09 | FairFace | ViT | 2 classes (gender), 5 of the 7 race groups | faces of the other 2 groups, East Asian and Southeast Asian (same classes) | 27,984 / 7,313 / 30,000 / 32,401 | 343 MB |
| 10 | Common Voice | Wav2Vec | 2 classes (gender), US and UK English accents | speakers with other English accents (Australian, Canadian, Scottish) | 70,578 / 9,951 / 29,556 / 42,697 | 416 MB |

Loading details (implemented in [experiments/use_cases.py](../experiments/use_cases.py)):
- UC04 and UC08: the drifted samples have the labels of the training classes.
- UC05: the drifted samples of both new digits have the label 8 in the file.
- UC08: the drifted windows at 5 % and 10 % drift are sampled entirely from `drift_05.hdf5` and `drift_10.hdf5`.
- UC09: the drifted samples keep the gender labels in the file; they are relabeled as a separate class (2) when
  loaded, and the embeddings are cast to float32.

## Source datasets and licenses

The archive contains embeddings and labels, not the original images or audio; the files of the text use cases
(UC01–UC03) also contain the input texts. The embeddings are derived from the following datasets; check the terms
of each one before reusing them.

| Dataset | Source | Terms |
|---|---|---|
| AG News | AG's corpus of news articles (Zhang et al., *Character-level Convolutional Networks for Text Classification*, 2015) | provided for research and other non-commercial use |
| 20 Newsgroups | http://qwone.com/~jason/20Newsgroups/ | no explicit license; public research corpus |
| Bias in Bios | De-Arteaga et al., *Bias in Bios*, 2019 (https://github.com/microsoft/biosbias) | see the source repository |
| MNIST | http://yann.lecun.com/exdb/mnist/ | CC BY-SA 3.0 |
| Intel Image Classification | https://www.kaggle.com/datasets/puneet6060/intel-image-classification | see the dataset page |
| STL-10 | https://cs.stanford.edu/~acoates/stl10/ | images from ImageNet; see the ImageNet terms of access |
| FairFace | https://github.com/joojs/fairface | CC BY 4.0 |
| Common Voice | https://commonvoice.mozilla.org | CC0 1.0 |
