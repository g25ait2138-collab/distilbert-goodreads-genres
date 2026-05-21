"""
data.py
--------
Download → sample → split → encode the UCSD Goodreads reviews dataset.

This script:
    1. Streams gzipped JSON review files for each of 8 genres from the
       UCSD McAuley Lab CDN.
    2. Samples 1,000 reviews per genre.
    3. Splits 800 / 200 (train / test) per genre, giving 6,400 train
       and 1,600 test reviews.
    4. Tokenises the texts with `DistilBertTokenizerFast`.
    5. Pickles the encoded train/test datasets and label maps to
       ./artifacts/data.pkl so train.py and eval.py can reload them.

Run directly from the command line:

    python data.py

Designed to be import-safe: nothing runs on `import`, everything is gated
by `if __name__ == "__main__":`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import random
from pathlib import Path

import requests
from transformers import DistilBertTokenizerFast

from utils import (
    GENRE_URL_DICT,
    MyDataset,
    build_label_maps,
)


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

DEFAULT_MODEL_NAME = "distilbert-base-cased"
DEFAULT_MAX_LENGTH = 512
DEFAULT_HEAD       = 10_000   # how many reviews to stream per genre
DEFAULT_SAMPLE     = 1_000    # how many reviews to keep per genre
DEFAULT_TRAIN_PER  = 800      # train reviews per genre
DEFAULT_SEED       = 42

ARTIFACTS_DIR = Path("./artifacts")
DATA_PICKLE   = ARTIFACTS_DIR / "data.pkl"


# ----------------------------------------------------------------------
# Streaming loader
# ----------------------------------------------------------------------

def load_reviews(url: str, head: int = DEFAULT_HEAD, sample_size: int = DEFAULT_SAMPLE):
    """
    Stream a gzipped JSON-lines file from `url`, take the first `head`
    review texts, then randomly sub-sample down to `sample_size`.
    """
    reviews = []
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with gzip.open(response.raw, "rt", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if i >= head:
                break
            d = json.loads(line)
            reviews.append(d["review_text"])

    if sample_size < len(reviews):
        reviews = random.sample(reviews, sample_size)
    return reviews


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def build_datasets(
    model_name:  str = DEFAULT_MODEL_NAME,
    max_length:  int = DEFAULT_MAX_LENGTH,
    head:        int = DEFAULT_HEAD,
    sample_size: int = DEFAULT_SAMPLE,
    train_per_genre: int = DEFAULT_TRAIN_PER,
    seed:        int = DEFAULT_SEED,
):
    """
    Run the full data pipeline and return everything downstream scripts need.
    """
    random.seed(seed)

    # 1. download + sample reviews per genre
    genre_reviews_dict = {}
    for genre, url in GENRE_URL_DICT.items():
        print(f"[data] loading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url, head=head, sample_size=sample_size)

    # 2. train / test split (per genre, fixed ratio)
    train_texts,  train_labels  = [], []
    test_texts,   test_labels   = [], []
    for genre, reviews in genre_reviews_dict.items():
        for r in reviews[:train_per_genre]:
            train_texts.append(r)
            train_labels.append(genre)
        for r in reviews[train_per_genre:]:
            test_texts.append(r)
            test_labels.append(genre)

    print(
        f"[data] train: {len(train_texts)} reviews | "
        f"test: {len(test_texts)} reviews"
    )

    # 3. label encoding
    label2id, id2label = build_label_maps(train_labels)
    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded  = [label2id[y] for y in test_labels]

    # 4. tokenisation
    print(f"[data] loading tokenizer: {model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    print("[data] tokenising train set...")
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=max_length
    )
    print("[data] tokenising test set...")
    test_encodings  = tokenizer(
        test_texts,  truncation=True, padding=True, max_length=max_length
    )

    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset  = MyDataset(test_encodings,  test_labels_encoded)

    return {
        "train_dataset": train_dataset,
        "test_dataset":  test_dataset,
        "label2id":      label2id,
        "id2label":      id2label,
        "tokenizer":     tokenizer,
        "train_texts":   train_texts,
        "test_texts":    test_texts,
        "train_labels":  train_labels,
        "test_labels":   test_labels,
    }


def save_artifacts(bundle: dict, path: Path = DATA_PICKLE):
    """Pickle the train/test datasets and label maps for train.py / eval.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    to_save = {
        "train_dataset": bundle["train_dataset"],
        "test_dataset":  bundle["test_dataset"],
        "label2id":      bundle["label2id"],
        "id2label":      bundle["id2label"],
        "test_labels":   bundle["test_labels"],
    }
    with open(path, "wb") as fp:
        pickle.dump(to_save, fp)
    print(f"[data] saved encoded datasets -> {path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Build the Goodreads train/test datasets.")
    p.add_argument("--model_name",     default=DEFAULT_MODEL_NAME)
    p.add_argument("--max_length",     type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--head",           type=int, default=DEFAULT_HEAD)
    p.add_argument("--sample_size",    type=int, default=DEFAULT_SAMPLE)
    p.add_argument("--train_per_genre",type=int, default=DEFAULT_TRAIN_PER)
    p.add_argument("--seed",           type=int, default=DEFAULT_SEED)
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    bundle = build_datasets(
        model_name=args.model_name,
        max_length=args.max_length,
        head=args.head,
        sample_size=args.sample_size,
        train_per_genre=args.train_per_genre,
        seed=args.seed,
    )
    save_artifacts(bundle)
    print("[data] done.")
