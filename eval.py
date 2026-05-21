"""
eval.py
--------
Final evaluation script. Loads the best fine-tuned model produced by
train.py, evaluates it on the held-out test set, logs the final loss /
accuracy / F1 / precision / recall to W&B, and saves the full per-class
classification report as a versioned W&B Artifact.

Run AFTER `python train.py`:

    python eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import torch
import wandb
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from utils import compute_metrics


DATA_PICKLE       = Path("./artifacts/data.pkl")
DEFAULT_MODEL_DIR = "./artifacts/best_model"
DEFAULT_REPORT    = Path("./artifacts/eval_report.json")
DEFAULT_WANDB_PROJECT  = "mlops-assignment2"
DEFAULT_WANDB_RUN_NAME = "distilbert-eval"


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_data():
    if not DATA_PICKLE.exists():
        raise FileNotFoundError(
            f"{DATA_PICKLE} not found. Run `python data.py` first."
        )
    with open(DATA_PICKLE, "rb") as fp:
        return pickle.load(fp)


def evaluate(args):
    # 1. Reload the encoded datasets and label maps
    bundle = _load_data()
    test_dataset = bundle["test_dataset"]
    id2label     = bundle["id2label"]

    # 2. Load the fine-tuned model.
    #    Prefer the local best-model dir; fall back to the HF Hub repo.
    if Path(args.model_dir).exists():
        print(f"[eval] loading model from local dir: {args.model_dir}")
        model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    else:
        print(f"[eval] loading model from HF Hub: {args.hf_repo_id}")
        model = DistilBertForSequenceClassification.from_pretrained(args.hf_repo_id)

    model.to(_device())

    # 3. Re-create a minimal Trainer just for evaluate() / predict()
    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=32,
        report_to="wandb",
        run_name=args.wandb_run_name,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 4. Start a fresh W&B run for evaluation
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        job_type="eval",
        resume="allow",
    )

    # 5. Run evaluation
    print("[eval] running trainer.evaluate()...")
    eval_results = trainer.evaluate()
    print("[eval] results:", eval_results)

    # 6. Log final metrics to W&B explicitly
    wandb.log(
        {
            "final/loss":      eval_results.get("eval_loss"),
            "final/accuracy":  eval_results.get("eval_accuracy"),
            "final/f1":        eval_results.get("eval_f1"),
            "final/precision": eval_results.get("eval_precision"),
            "final/recall":    eval_results.get("eval_recall"),
        }
    )

    # 7. Build and save the full classification report
    preds  = trainer.predict(test_dataset).predictions.argmax(-1)
    labels = [item["labels"].item() for item in test_dataset]
    target_names = [id2label[i] for i in sorted(id2label)]

    report = classification_report(
        labels,
        preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report["_summary_metrics"] = eval_results

    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"[eval] wrote classification report -> {args.report_path}")

    # 8. Upload as a versioned W&B Artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(args.report_path)
    wandb.log_artifact(artifact)
    print("[eval] uploaded W&B artifact: eval-report")

    # 9. Record the HF model URL in this run's summary too
    wandb.run.summary["huggingface_model"] = (
        f"https://huggingface.co/{args.hf_repo_id}"
    )

    wandb.finish()
    print("[eval] done.")


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate the fine-tuned model.")
    p.add_argument("--model_dir",      default=DEFAULT_MODEL_DIR)
    p.add_argument("--hf_repo_id",     default="ALOK2026/distilbert-goodreads-genres")
    p.add_argument("--report_path",    default=str(DEFAULT_REPORT))
    p.add_argument("--wandb_project",  default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--wandb_run_name", default=DEFAULT_WANDB_RUN_NAME)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
