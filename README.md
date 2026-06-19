# Toxic Comment Classification

> Binary toxic comment classifier trained on the Jigsaw Toxic Comment Classification dataset, comparing a TF-IDF + Logistic Regression baseline against a fine-tuned RoBERTa transformer.

This project tackles the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge), originally a multi-label problem with six toxicity categories (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`). Here it's simplified to a binary task: a comment is labeled toxic (`1`) if it's flagged under *any* of those six categories, and non-toxic (`0`) otherwise. The goal is to compare how a classical sparse-feature model stacks up against a modern transformer on the same data.

## Project structure

```
.
├── notebooks/
│   ├── 01_eda.ipynb                  # Class balance, comment length, word clouds
│   ├── 02_baseline_model.ipynb       # TF-IDF + Logistic Regression baselines
│   └── 03_roberta_finetuning.ipynb   # Fine-tuned roberta-base classifier
├── data/                             # train.csv goes here (see data/README.md)
├── models/                           # Trained model saved here (see models/README.md)
├── requirements.txt
└── README.md
```

## Approach

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
The dataset is heavily imbalanced — only around 10% of comments are labeled toxic, which shapes every modeling decision downstream. The EDA looks at the distribution across the six original label categories (toxic comments often carry multiple overlapping labels, e.g. `obscene` + `insult` together), compares comment length between toxic and non-toxic classes (toxic comments tend to run longer), and uses word clouds to surface the vocabulary that's most distinctive to toxic comments versus the overall corpus.

### 2. Baseline model (`02_baseline_model.ipynb`)
Text is vectorized with TF-IDF (10,000-word vocabulary) and fed into Logistic Regression, the standard strong baseline for text classification. Three variants are trained to handle the class imbalance differently:
- **Plain Logistic Regression** — no imbalance handling, trained directly on the skewed data.
- **Class-weight balanced** — `class_weight="balanced"` upweights the minority (toxic) class during training.
- **Random undersampling** — the majority (non-toxic) class is downsampled so the model sees a roughly even split.

Comparing these three isolates the precision/recall trade-off: balancing strategies push recall up (catching more toxic comments) at the cost of precision (more false positives), which matters for real-world moderation where both error types have different costs.

### 3. RoBERTa fine-tuning (`03_roberta_finetuning.ipynb`)
`roberta-base` is fine-tuned end-to-end using the Hugging Face `Trainer` API. Training setup: 5 epochs, learning rate 2e-5, with early stopping based on validation F1 to avoid overfitting once performance plateaus. Unlike the TF-IDF baseline, RoBERTa's contextual embeddings capture word order, negation, and sarcasm-adjacent phrasing that bag-of-words features miss entirely — which is the core hypothesis being tested against the baseline.

## Results

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.92 | 0.63 | 0.74 |
| Logistic Regression (class-weight balanced) | 0.62 | 0.88 | 0.73 |
| Logistic Regression (random undersampling) | 0.55 | 0.88 | 0.68 |
| **RoBERTa (fine-tuned)** | **0.84** | **0.84** | **0.84** |

The plain Logistic Regression model is precision-heavy but misses a lot of toxic comments (low recall) — it defaults toward predicting "non-toxic" because that's the majority class. The balancing techniques fix recall but at a steep precision cost, flagging far more false positives. RoBERTa is the only model that achieves a strong score on *both* metrics simultaneously, because it doesn't need an artificial rebalancing trick to find the minority class — the contextual representations make toxic language separable from the embeddings themselves, rather than relying on sampling tricks at the data level.

## License

MIT
