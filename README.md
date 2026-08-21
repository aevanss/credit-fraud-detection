# Credit Card Fraud Detection

EDA and Predictive Modeling project to determine optimal way to detect fraudulent credit card transactions from transaction data history.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations & Future Work](#limitations--future-work)
- [Tech Stack](#tech-stack)

## Overview

This project builds a model to detect fraudulent credit card transactions in the most accurate way possible, enough to catch real fraud without being so aggressive that it floods customers with false alarms, or so conservative that fraud slips through undetected. The core challenge is severe class imbalance: only 0.17% of transactions in this dataset are fraudulent, meaning a naive model that predicts "not fraud" every single time would still score ~99.8% accuracy while catching no actual fraud. Most features in the dataset are PCA-anonymized (V1-V28), transformed before release by the original dataset creators to protect sensitive customer data. This means the model has to work without direct domain interpretability on those columns. Fraud and fraud detection are also constantly changing, so a model trained on a fixed historical window like this one reflects a snapshot rather than a permanent solution (concept drift).

This project trains and compares a logistic regression baseline against a threshold-tuned Random Forest classifier, ultimately selecting Random Forest (PR-AUC 0.814) as the final model — catching ~81% of fraud in the test set while maintaining 84% precision.

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 columns (28 anonymized PCA components `V1`-`V28`, plus `Time`, `Amount`, `Class`)
- **Class balance:** 0.17% fraud rate (492 fraud cases out of 284,807 transactions)
- **Why anonymized:** The original dataset creators (ULB) applied PCA to the raw transaction features before release, specifically to protect sensitive customer data. This means the model has no domain-level interpretability on `V1`-`V28` — feature importance can show *which* components matter, but not *why*, in business terms.
- Not included in this repo (~144MB, gitignored) — download and place at `data/raw/creditcard.csv`. See Setup below.

## Project Structure

credit_fraud/
├── data/raw/ # not tracked — see Setup
├── notebooks/
│ └── eda.ipynb
├── venv/ # not tracked — recreate via Setup
├── .gitignore
├── requirements.txt
├── SETUP_NOTES.md # environment setup + troubleshooting log
└── README.md

## Setup

```bash
git clone <repo-url>
cd credit_fraud
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Kaggle API credentials are required to download the dataset — see [`SETUP_NOTES.md`](./SETUP_NOTES.md) for credential setup and known environment issues (Python version constraints, Kaggle auth).

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

## Methodology

### 1. Data Cleaning
Found 1,081 exact duplicate rows that were identical across all 31 columns, including 28 continuous, high-precision PCA float values. Given the near-zero probability of two different transactions matching precisely by chance, these were treated as a data pipeline artifact (e.g., duplicate logging) rather than real distinct events, and dropped (`284,807` → `283,726` rows). No missing values were found in any column.

### 2. Feature Engineering
Derived `hour_of_day` from `Time` (`Time % 86400 // 3600`), converting elapsed seconds into a 0-23 hour-of-day bucket. EDA showed fraud rate spiking to ~1.7% at hour 2 (roughly 10x the dataset's overall 0.17% base rate) suggesting real, non-random temporal structure in fraud occurrence.

### 3. Train/Test Split
80/20 split, stratified on `Class` to preserve the ~0.17% fraud rate identically in both sets, preventing a random split from under- or over-representing the already rare fraud class in either partition.

### 4. Scaling
Applied `RobustScaler` (median/IQR-based, rather than mean/std) to all features, fit exclusively on training data and applied to both train and test to prevent leakage. Robust scaling was chosen after `.skew()` revealed significant skew in `Amount` and a handful of PCA components (notably `V28`, skew ≈ 12.2, and `V8`, skew ≈ -8.0), overturning an initial assumption that PCA output would already be well-behaved.

### 5. Modeling
- **Baseline:** `DummyClassifier` (always predicts the majority class) — establishes the ~99.8% accuracy floor any real model must be judged against, not with, given the severity of the class imbalance.
- **Logistic Regression** (`class_weight='balanced'`) — PR-AUC 0.678.
- **Random Forest** (`class_weight='balanced'`, 100 trees) — PR-AUC 0.814, and its full precision-recall curve sat above logistic regression's at nearly every recall level, not just on aggregate PR-AUC.

### 6. Threshold Tuning
Default classification thresholds (0.5) are not necessarily optimal for an imbalanced, cost-asymmetric problem like fraud detection, where a missed fraud case (false negative) costs real money while a false positive costs customer friction. Thresholds were swept from 0.1 to 0.9 for both models, and precision/recall were evaluated at each point rather than relying on the default cutoff. For logistic regression, recall remained flat (0.874) from threshold 0.3 through 0.7 while precision nearly doubled over that range — making 0.7 a "free" improvement with no recall cost. Random Forest showed strong precision (0.837+) even at its most lenient tested threshold (0.1), leading to a final choice of threshold 0.1 to maximize fraud recall while retaining strong precision.

## Results

| Model | Threshold | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7 | 0.10 | 0.87 | 0.18 | 0.678 |
| Random Forest | 0.1 | 0.84 | 0.81 | 0.82 | 0.814 |

**Chosen model: Random Forest, threshold 0.1.** Random Forest's precision-recall curve dominated logistic regression's across nearly the entire recall range, meaning the improvement is consistent, not simply an average. At the chosen threshold, the model catches approximately 77 of 95 fraud cases in the test set (81% recall) while maintaining 84% precision, meaning roughly 5 out of every 6 fraud alerts raised are actually fraud.

## Key Findings

- **Accuracy is meaningless here.** With a 0.17% fraud rate, a model that predicts "not fraud" for every transaction scores ~99.8% accuracy while catching zero fraud. Precision, recall, F1, and PR-AUC were used throughout instead.
- **Random Forest substantially outperformed logistic regression**, suggesting fraud patterns in this dataset involve non-linear relationships and feature interactions that a linear model structurally cannot capture.
- **Feature importance partially, but not fully, overlaps with linear correlation.** Random Forest's top features (`V14`, `V10`, `V12`, `V4`, `V11`) share several entries with the top linear-correlation features (`V17`, `V14`, `V12`, `V10`, `V16`), but the rankings differ meaningfully — `V14` ranked far higher in feature importance (0.197) than in correlation magnitude, suggesting it carries predictive signal that a purely linear measure understates.
- **`hour_of_day` showed a real EDA pattern but low model importance.** Fraud rate spiked ~10x the baseline at hour 2, yet the engineered `hour_of_day` feature ranked near the bottom of Random Forest's feature importance (0.004, out of 31 features). A likely explanation: the anonymized `V` columns may already capture similar timing information, since they were derived from the original transaction data. That makes `hour_of_day` less useful once those features are already available to the model.

## Limitations & Future Work

- **Room for improvement remains.** At the chosen threshold, the model still misses ~19% of real fraud cases (18 of 95 in the test set) — a meaningful gap for a production fraud system, even with Random Forest's clear improvement over the logistic regression baseline.
- **Limited explainability.** Because `V1`-`V28` were PCA-transformed by the original dataset creators for anonymization, feature importance can show *which* components matter most, but not *why*, in business terms — a real constraint for any deployment context that requires explaining individual fraud flags (e.g., to regulators or customers).
- **No handling of concept drift.** This model reflects patterns present in a fixed ~48-hour historical window; real fraud tactics evolve over time, and a production system would need periodic retraining or drift monitoring.

**Future work:**
- Compare against additional non-linear/ensemble models (XGBoost, LightGBM), given Random Forest's clear outperformance of the linear baseline here.
- Try SMOTE as an alternative to `class_weight='balanced'` for handling class imbalance.
- Mark the chosen operating threshold directly on the precision-recall curve plot, making the final decision point visually explicit rather than requiring a cross-reference to the results table.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, imbalanced-learn, Jupyter