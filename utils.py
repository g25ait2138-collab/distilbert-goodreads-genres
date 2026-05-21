"""
utils.py
---------
Shared helpers for the MLOps Assignment 2 pipeline.

Contains:
    * GENRES                   : canonical list of target genre labels
    * GENRE_URL_DICT           : mapping {genre -> UCSD Goodreads .json.gz URL}
    * MyDataset                : a torch.utils.data.Dataset wrapper around
                                 HuggingFace tokenizer encodings + integer labels
    * build_label_maps()       : returns (label2id, id2label) given a list of labels
    * compute_metrics()        : passed to the HuggingFace Trainer; returns
                                 accuracy, weighted F1, weighted precision,
                                 and weighted recall
"""

from __future__ import annotations

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ----------------------------------------------------------------------
# Dataset constants
# ----------------------------------------------------------------------

GENRES = [
    "poetry",
    "children",
    "comics_graphic",
    "fantasy_paranormal",
    "history_biography",
    "mystery_thriller_crime",
    "romance",
    "young_adult",
]

# UCSD Goodreads "by genre" review dumps (gzipped JSON).
# Source: https://mengtingwan.github.io/data/goodreads.html
_BASE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
)
GENRE_URL_DICT = {
    genre: f"{_BASE_URL}goodreads_reviews_{genre}.json.gz" for genre in GENRES
}


# ----------------------------------------------------------------------
# Torch Dataset
# ----------------------------------------------------------------------

class MyDataset(torch.utils.data.Dataset):
    """
    Simple Dataset that pairs HuggingFace tokenizer encodings with integer labels.

    Parameters
    ----------
    encodings : transformers.BatchEncoding
        Output of `tokenizer(texts, truncation=True, padding=True, ...)`.
    labels : list[int]
        Integer-encoded class labels, one per text.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ----------------------------------------------------------------------
# Label maps
# ----------------------------------------------------------------------

def build_label_maps(labels):
    """
    Build {label -> id} and {id -> label} dictionaries from a list of string labels.

    The ordering is deterministic (sorted) so the mapping is reproducible
    across runs and machines.
    """
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ----------------------------------------------------------------------
# Trainer compute_metrics
# ----------------------------------------------------------------------

def compute_metrics(pred):
    """
    Function passed to `transformers.Trainer(compute_metrics=...)`.

    Returns weighted-averaged accuracy, F1, precision and recall so we get
    a single comparable number per epoch even when the test set is balanced.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall":    recall_score(labels, preds, average="weighted"),
    }
