# NBA Draft Prediction 🏀

A machine learning project that predicts whether a college basketball player will be selected in the NBA Draft, using historical player statistics. Built as part of the Applied Machine Learning & AI (AMLA) subject in the Master of Data Science & Innovation program at UTS Sydney.

## Problem Statement

**Task:** Binary classification — will a college basketball player be drafted into the NBA?

**Challenge:** Severe class imbalance. Only ~0.8% of college players get drafted (94 positives out of ~11,800 players). Standard accuracy is misleading — AUROC and Top-K recall are the meaningful metrics here.

**Why it matters:** NBA teams spend millions on draft picks. A model that can reliably surface high-potential players from thousands of candidates reduces scouting bias and improves team decision-making.

## Results

| Experiment | Model | AUROC | PR-AUC | Top-200 Recall |
|---|---|---|---|---|
| 1 | XGBoost (all features) | **0.9961** | 0.6633 | 1.00 |
| 2 | XGBoost (curated domain features) | **0.9968** | 0.7032 | 1.00 |
| 3 | Random Forest | 0.9945 | 0.4901 | 1.00 |
| 4 | CatBoost | — | — | — |

**Best model:** XGBoost with curated domain features (Experiment 2) — AUROC 0.9968, PR-AUC 0.7032.

All 4 experiments confirmed the hypothesis that ML can effectively rank players by draft likelihood.

## Dataset

- Source: [Kaggle NBA Draft Prediction competition](https://www.kaggle.com/)
- Train set: ~14,700 rows, ~12,600 features (after one-hot encoding)
- Target: `drafted` (binary — 1 = drafted, 0 = not drafted)
- Key features: points per game, assists, rebounds, shooting efficiency, advanced metrics (BPM, VORP, etc.), recruitment rank

## Experiments

| Notebook | Description |
|---|---|
| `Experiment_1_XGBoost_All_Features.ipynb` | Baseline XGBoost using all available features. Feature selection via correlation + RandomForest importance. |
| `Experiment_2_XGBoost_Curated_Features.ipynb` | XGBoost with hand-crafted domain features (per-minute ratios, efficiency metrics, advanced ratings). Highest PR-AUC. |
| `Experiment_3_Random_Forest.ipynb` | Random Forest with balanced subsampling to handle class imbalance. |
| `Experiment_4_CatBoost.ipynb` | CatBoost with native categorical handling. Depth=6, learning_rate=0.05, early stopping. |

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

> **Note:** Dataset must be downloaded separately from Kaggle. See the dataset section in each notebook for setup instructions.

## Key Findings

- XGBoost outperforms Random Forest and CatBoost on this dataset
- Curated domain features (per-minute stats, efficiency metrics) improve PR-AUC from 0.663 to 0.703 over raw features
- Top-200 recall of 1.0 across all models — every drafted player is captured in the top 200 predictions
- Class imbalance (124:1 ratio) requires careful handling; `scale_pos_weight` in XGBoost/CatBoost is critical

---

*Built with XGBoost · CatBoost · scikit-learn · wandb · LIME*
