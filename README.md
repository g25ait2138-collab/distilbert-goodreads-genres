# Fine-Tuning DistilBERT for Goodreads Genre Classification

> **MLOps Assignment 2 — PGD AI Programme, IIT Jodhpur**
> End-to-end MLOps workflow: starter notebook → modular Python scripts → fine-tuning on **Kaggle Notebook (GPU)** → experiment tracking with **Weights & Biases** → model publishing on the **Hugging Face Hub**.

---

## Project Description

This project fine-tunes a pre-trained **DistilBERT** (`distilbert-base-cased`) transformer on the **UCSD Goodreads** book-reviews dataset to classify reviews into one of eight book genres. The starter Colab notebook is reorganised into four production-grade Python modules (`data.py`, `train.py`, `eval.py`, `utils.py`); every training run is tracked end-to-end with Weights & Biases (loss curves, accuracy, F1, learning-rate schedule, GPU utilisation, hyperparameters); the final fine-tuned model is evaluated on a held-out test split and the classification report is uploaded as a versioned W&B Artifact; and both the model weights and tokenizer are published to a public Hugging Face repository so anyone can load and use the model with two lines of code.

The focus of this assignment is the **MLOps workflow** around the model rather than the model's internals.

### Target Genres

- Children
- Comics & Graphic
- Fantasy & Paranormal
- History & Biography
- Mystery / Thriller / Crime
- Poetry
- Romance
- Young Adult

---

## Training Platform

This project was trained on a **Kaggle Notebook** using a free Kaggle GPU (NVIDIA Tesla T4). API tokens (`WANDB_API_KEY`, `HF_TOKEN`) were stored as **Kaggle Secrets** (Add-ons → Secrets) and read at runtime via `kaggle_secrets.UserSecretsClient`, so they never appear in the notebook source.

| Component   | Details                          |
|-------------|----------------------------------|
| Platform    | Kaggle Notebook                  |
| GPU         | NVIDIA Tesla T4 (free tier)      |
| Framework   | Hugging Face Transformers        |
| Backend     | PyTorch                          |
| Language    | Python 3.10+                     |

---

## Project Structure

```
.
├── data.py             # Data loading, sampling, train/test split, label encoding
├── train.py            # Model loading, Trainer setup, training loop, W&B logging, push to HF Hub
├── eval.py             # Final evaluation, metrics, classification report -> W&B Artifact
├── utils.py            # Shared helpers (label maps, Dataset class, compute_metrics)
├── requirements.txt    # Pinned dependencies
├── README.md           # You are here
└── report.pdf          # Final report (3 pages)
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-github-username>/mlops-assignment2.git
cd mlops-assignment2
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API tokens

```bash
export HF_TOKEN="<your_huggingface_token>"
export WANDB_API_KEY="<your_wandb_api_key>"
```

> Get a free Hugging Face token at <https://huggingface.co/settings/tokens> and a W&B key at <https://wandb.ai/authorize>.

These tokens are required for:

- Experiment tracking with Weights & Biases
- Model uploading to the Hugging Face Hub

---

## How to Run

The scripts run sequentially from the command line:

```bash
# 1. Download UCSD Goodreads reviews, sample, split, encode, and pickle to ./artifacts/data.pkl
python data.py

# 2. Fine-tune DistilBERT, log to W&B, and push the model + tokenizer to the HF Hub
python train.py

# 3. Run final evaluation, log metrics to W&B, save the classification report as a W&B Artifact
python eval.py
```

> **GPU note.** Training was run on a Kaggle T4 GPU and takes roughly 9–10 minutes for 3 epochs. On CPU, reduce `--sample_size` to approximately 200 per genre and expect significantly longer runtimes.

---

## Training Configuration

| Parameter         | Value                  |
|-------------------|------------------------|
| Base Model        | `distilbert-base-cased`|
| Epochs            | 3                      |
| Batch Size        | 16                     |
| Learning Rate     | 3e-5                   |
| Max Sequence Length | 128 tokens           |
| Dataset           | UCSD Goodreads Reviews |
| Training Samples  | 6,400                  |
| Test Samples      | 1,600                  |
| Best-Model Strategy | `load_best_model_at_end=True` (lowest validation loss) |

---

## Results

### Final Test-Set Metrics

| Metric                | Score   |
|-----------------------|---------|
| Accuracy              | 0.5619  |
| F1 Score (weighted)   | 0.5673  |
| Precision (weighted)  | 0.5781  |
| Recall (weighted)     | 0.5619  |
| Evaluation Loss       | 2.8248  |

### Per-Epoch Training Progress

| Epoch | Training Loss | Validation Loss | Accuracy | F1     | Precision | Recall |
|-------|---------------|-----------------|----------|--------|-----------|--------|
| 1     | 1.0665        | 2.8248          | 0.5619   | 0.5673 | 0.5781    | 0.5619 |
| 2     | 0.9481        | 3.0193          | 0.5625   | 0.5674 | 0.5792    | 0.5625 |
| 3     | 0.5708        | 3.1652          | 0.5713   | 0.5747 | 0.5792    | 0.5713 |

> The epoch-1 checkpoint was selected automatically based on the lowest validation loss.

### Per-Class Classification Report (test set, 200 reviews/genre)

| Class                    | Precision | Recall | F1   |
|--------------------------|-----------|--------|------|
| children                 | 0.64      | 0.60   | 0.62 |
| comics_graphic           | 0.88      | 0.75   | 0.81 |
| fantasy_paranormal       | 0.36      | 0.34   | 0.35 |
| history_biography        | 0.54      | 0.53   | 0.53 |
| mystery_thriller_crime   | 0.55      | 0.59   | 0.57 |
| poetry                   | 0.71      | 0.76   | 0.73 |
| romance                  | 0.64      | 0.52   | 0.57 |
| young_adult              | 0.31      | 0.41   | 0.35 |
| **weighted avg**         | **0.58**  | **0.56** | **0.57** |

### Baseline Comparison

| Model                         | Accuracy | Weighted F1 |
|-------------------------------|----------|-------------|
| Logistic Regression (TF-IDF)  | 0.53     | 0.53        |
| **DistilBERT (fine-tuned)**   | **0.56** | **0.57**    |

The fine-tuned DistilBERT model outperforms the traditional TF-IDF + Logistic Regression baseline on both accuracy and weighted F1.

---

## Quick Use — Load the Published Model

Once the model is on the Hugging Face Hub, anyone can load it with a few lines of code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo_id = "ALOK2026/distilbert-goodreads-genres"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model     = AutoModelForSequenceClassification.from_pretrained(repo_id)

inputs = tokenizer(
    "A thrilling whodunit set in Victorian London.",
    return_tensors="pt", truncation=True, max_length=512,
)
pred_id = model(**inputs).logits.argmax(-1).item()
print(model.config.id2label[pred_id])
```

---

## Public Resources

- **Kaggle Notebook:** <https://www.kaggle.com/code/aloksahnig25ait2138/notebook8d09f3f76d/script>
- **Hugging Face Model:** <https://huggingface.co/ALOK2026/distilbert-goodreads-genres >
- **W&B Project Dashboard:** <https://wandb.ai/g25ait2138-iitj/mlops-assignment2/runs/0ytv9ju6?nw=nwuserg25ait2138>
- **GitHub Repository: https://github.com/g25ait2138-collab/distilbert-goodreads-genres 

---

## Tech Stack

- **Modeling:** PyTorch · Hugging Face Transformers · DistilBERT (`distilbert-base-cased`)
- **Experiment Tracking:** Weights & Biases
- **Model Hosting:** Hugging Face Hub
- **Training Platform:** Kaggle Notebook (free-tier GPU)
- **Dataset:** UCSD Goodreads book reviews
- **Language:** Python 3.10+

---

## Conclusion

This project successfully demonstrates an end-to-end transformer-based NLP classification pipeline integrated with modern MLOps practices. The implementation covers scalable training, cloud-based GPU execution, live experiment monitoring, reproducibility through modular scripts, and public deployment of trained artifacts.

The final fine-tuned DistilBERT model achieves a measurable improvement over the TF-IDF + Logistic Regression baseline while keeping the training and deployment workflow lightweight, reproducible, and fully tracked.

---


