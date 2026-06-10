# NBA Draft Prediction 🏀

A machine learning project that predicts whether a college basketball player will be selected in the NBA Draft, using historical player statistics.

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://www.kaggle.com/competitions/uts-36120-25-sp/overview)
![Score](https://img.shields.io/badge/Kaggle%20Score-0.99610%20AUROC-brightgreen)
![Rank](https://img.shields.io/badge/Rank-37%2F61%20Teams-orange)

Built as part of the Applied Machine Learning & AI (AMLA) subject in the Master of Data Science & Innovation program at UTS Sydney.

## Kaggle Competition

| Detail | Value |
|---|---|
| Competition | [UTS-36120-25SP](https://www.kaggle.com/competitions/uts-36120-25-sp/overview) |
| Final Score (AUROC) | **0.99610** |
| Final Rank | **37 / 61 teams** |
| Total Entrants | 144 |
| Total Submissions | 1,557 |
| Evaluation Metric | ROC-AUC |

## Problem Statement

**Task:** Binary classification — will a college basketball player be drafted into the NBA?

**Challenge:** Severe class imbalance. Only ~0.8% of college players get drafted (94 positives out of ~11,800 players). Standard accuracy is misleading — AUROC and Top-K recall are the meaningful metrics here.

**Why it matters:** NBA teams spend millions on draft picks. A model that reliably surfaces high-potential players from thousands of candidates reduces scouting bias and improves team decision-making.

## Experiment Results

| Experiment | Model | Validation AUROC | PR-AUC | Top-200 Recall |
|---|---|---|---|---|
| 1 | XGBoost (all features) | 0.9961 | 0.6633 | 1.00 |
| 2 | XGBoost (curated domain features) | **0.9968** | **0.7032** | 1.00 |
| 3 | Random Forest | 0.9945 | 0.4901 | 1.00 |
| 4 | CatBoost | — | — | — |

All 4 experiments confirmed the hypothesis. Best model: XGBoost with curated domain features (Exp 2).

## Dataset

- Source: [Kaggle — UTS-36120-25SP](https://www.kaggle.com/competitions/uts-36120-25-sp/overview)
- Train set: ~14,700 rows, ~12,600 features (after one-hot encoding)
- Target: `drafted` (binary — 1 = drafted, 0 = not drafted)
- Key features: points per game, assists, rebounds, shooting efficiency, advanced metrics (BPM, VORP), recruitment rank

## Experiments

| Notebook | Description |
|---|---|
| `Experiment_1_XGBoost_All_Features.ipynb` | Baseline XGBoost with all features. Feature selection via correlation + RandomForest importance. |
| `Experiment_2_XGBoost_Curated_Features.ipynb` | XGBoost with curated domain features (per-minute ratios, efficiency metrics, advanced ratings). Best PR-AUC. |
| `Experiment_3_Random_Forest.ipynb` | Random Forest with balanced subsampling to handle class imbalance. |
| `Experiment_4_CatBoost.ipynb` | CatBoost with native categorical handling. Depth=6, lr=0.05, early stopping. |

## Tech Stack

- **ML:** XGBoost, CatBoost, scikit-learn (Random Forest, Logistic Regression)
- **Experiment Tracking:** Weights & Biases (wandb)
- **Interpretability:** LIME
- **Data:** pandas, numpy
- **Visualisation:** matplotlib, seaborn, plotly

## Setup

```bash
git clone https://github.com/Mitesh2412/nba-draft-prediction.git
cd nba-draft-prediction

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab
```

> **Note:** Dataset must be downloaded from the [Kaggle competition page](https://www.kaggle.com/competitions/uts-36120-25-sp/overview). Requires Kaggle account.

## Key Findings

- XGBoost outperforms Random Forest and CatBoost on this dataset
- Curated domain features improve PR-AUC from 0.663 → 0.703 over raw features
- Top-200 recall of 1.0 across all models — every drafted player captured in top 200 predictions
- Class imbalance of 124:1 requires careful handling via `scale_pos_weight`

---

*Built with XGBoost · CatBoost · scikit-learn · wandb · LIME*
