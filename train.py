"""
train.py
---------
Fine-tune DistilBERT on the Goodreads reviews dataset and log everything
to Weights & Biases. After training, push the model + tokenizer to the
Hugging Face Hub and record the model URL in the W&B run summary.

Run AFTER `python data.py` has produced ./artifacts/data.pkl:

    python train.py

Required environment variables:
    WANDB_API_KEY  — get one at https://wandb.ai/authorize
    HF_TOKEN       — get one at https://huggingface.co/settings/tokens

On Kaggle, set these as Kaggle Secrets (Add-ons -> Secrets) and load with
`UserSecretsClient`; the notebook entry-cell does this already.

Hyperparameters (epochs, batch size, learning rate, max_length) are kept
identical to the starter Colab notebook so this script reproduces those
results when run on the same hardware.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import torch
import wandb
from huggingface_hub import login
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils import compute_metrics


# ----------------------------------------------------------------------
# Defaults (match the Kaggle notebook)
# ----------------------------------------------------------------------

DEFAULT_MODEL_NAME      = "distilbert-base-cased"
DEFAULT_EPOCHS          = 3
DEFAULT_TRAIN_BATCH     = 16
DEFAULT_EVAL_BATCH      = 32
DEFAULT_LEARNING_RATE   = 3e-5
DEFAULT_MAX_LENGTH      = 512
DEFAULT_WARMUP_STEPS    = 100
DEFAULT_WEIGHT_DECAY    = 0.01
DEFAULT_LOGGING_STEPS   = 50
DEFAULT_OUTPUT_DIR      = "./results"
DEFAULT_WANDB_PROJECT   = "mlops-assignment2"
DEFAULT_WANDB_RUN_NAME  = "distilbert-run-1"
DEFAULT_HF_REPO_ID      = "ALOK2026/distilbert-goodreads-genres"

DATA_PICKLE       = Path("./artifacts/data.pkl")
TRAINER_STATE_PKL = Path("./artifacts/trainer_state.pkl")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load_data():
    if not DATA_PICKLE.exists():
        raise FileNotFoundError(
            f"{DATA_PICKLE} not found. Run `python data.py` first."
        )
    with open(DATA_PICKLE, "rb") as fp:
        return pickle.load(fp)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

def train(args):
    # 1. Reload the encoded datasets
    bundle = _load_data()
    train_dataset = bundle["train_dataset"]
    test_dataset  = bundle["test_dataset"]
    id2label      = bundle["id2label"]
    label2id      = bundle["label2id"]

    device = _device()
    print(f"[train] using device: {device}")
    print(f"[train] num labels:   {len(id2label)}")

    # 2. Initialise W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model":        args.model_name,
            "epochs":       args.epochs,
            "batch_size":   args.train_batch_size,
            "learning_rate":args.learning_rate,
            "max_length":   args.max_length,
            "dataset":      "UCSD Goodreads",
            "platform":     "Kaggle",
        },
    )

    # 3. Load the pre-trained model with the correct number of output labels
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # 4. Training arguments — report_to="wandb" enables full W&B logging
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Train!
    print("[train] starting training loop...")
    trainer.train()

    # 7. Save the trainer state locally (eval.py will reload model + tokenizer
    #    directly from the HF Hub or from `output_dir/best_model`).
    TRAINER_STATE_PKL.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(TRAINER_STATE_PKL.parent / "best_model"))

    # 8. Push model + tokenizer to the Hugging Face Hub
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and args.push_to_hub:
        print(f"[train] pushing model to HF Hub: {args.hf_repo_id}")
        login(token=hf_token)
        model.push_to_hub(args.hf_repo_id)
        bundle["tokenizer"] = bundle.get("tokenizer")  # re-load lazily if needed
        # The tokenizer object was created in data.py and pickled together; if
        # it isn't there (e.g. legacy pickle), reload it from the base model.
        tokenizer = bundle.get("tokenizer")
        if tokenizer is None:
            from transformers import DistilBertTokenizerFast
            tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
        tokenizer.push_to_hub(args.hf_repo_id)

        # 9. Record the HF model URL in the W&B run summary
        hf_url = f"https://huggingface.co/{args.hf_repo_id}"
        wandb.run.summary["huggingface_model"] = hf_url
        print(f"[train] HF model URL: {hf_url}")
    else:
        print("[train] HF_TOKEN not set (or --no_push given); skipping push.")

    wandb.finish()
    print("[train] done.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT on Goodreads.")
    p.add_argument("--model_name",        default=DEFAULT_MODEL_NAME)
    p.add_argument("--epochs",            type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--train_batch_size",  type=int,   default=DEFAULT_TRAIN_BATCH)
    p.add_argument("--eval_batch_size",   type=int,   default=DEFAULT_EVAL_BATCH)
    p.add_argument("--learning_rate",     type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--max_length",        type=int,   default=DEFAULT_MAX_LENGTH)
    p.add_argument("--warmup_steps",      type=int,   default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--weight_decay",      type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--logging_steps",     type=int,   default=DEFAULT_LOGGING_STEPS)
    p.add_argument("--output_dir",        default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--wandb_project",     default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--wandb_run_name",    default=DEFAULT_WANDB_RUN_NAME)
    p.add_argument("--hf_repo_id",        default=DEFAULT_HF_REPO_ID)
    p.add_argument("--no_push", dest="push_to_hub", action="store_false")
    p.set_defaults(push_to_hub=True)
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
