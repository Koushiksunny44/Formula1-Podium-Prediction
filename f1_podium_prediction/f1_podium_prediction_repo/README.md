# F1 Podium Prediction — Supervised Learning Assignment

Predicting whether a Formula 1 driver will finish on the **podium** (top 3) of a Grand Prix using **only pre-race information** — starting grid position, weather, driver/constructor history, and circuit. Binary classification, 2000–2025 seasons.

> **Best result:** XGBoost — F1 = 0.655, AUC = 0.927 on held-out 2023–2025 seasons.

---

## Project Structure

```
f1_podium_prediction/
├── data/
│   └── f1_cleaned.xlsx                    # 10,398 rows × 20 cols
├── notebooks/
│   └── F1_Podium_Prediction.ipynb         # Executed notebook with outputs
├── results/
│   ├── Table_I_baseline_performance.csv
│   ├── Table_II_model_comparison.csv
│   ├── Table_III_preprocessing_ablation.csv
│   ├── Table_IV_feature_importance.csv
│   ├── Table_V_ranking_sensitivity.csv
│   ├── Table_VI_robustness.csv
│   ├── Table_VII_decision_matrix.csv
│   └── summary.json
├── figures/
│   ├── Figure_1_baseline_performance.png
│   ├── Figure_2_model_ranking.png
│   ├── Figure_2b_roc_curves.png
│   ├── Figure_3_preprocessing_ablation.png
│   ├── Figure_4_feature_importance.png
│   ├── Figure_5_ranking_sensitivity.png
│   ├── Figure_6_robustness_cv.png
│   ├── Figure_7_decision_radar.png
│   └── Figure_extra_confusion_matrix.png
├── f1_podium_analysis.py                  # Full pipeline as a script
├── requirements.txt
└── README.md
```

---

## Dataset

- **Name:** Formula 1 Race Results, 2000–2025
- **Source:** Cleaned subset of the public Ergast / Kaggle F1 dataset — see [Kaggle: Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- **Rows:** 10,398 driver-race entries
- **Columns:** 20 (8 numeric, 12 categorical/text)
- **Target:** `Podium` ∈ {Yes, No} — binary, ~14.3% positive (imbalanced)

### Pre-race features used (no leakage)

| Feature | Type | Description |
|---|---|---|
| Grid | numeric | Starting grid position |
| Rain | numeric | Wet/dry race indicator |
| Round | numeric | Race number within season |
| Season | numeric | Year |
| ConstructorName | categorical | Team |
| DriverNationality | categorical | Driver country |
| country | categorical | Race host country |
| DriverPodiumRate_Last5 | engineered | Driver's podium rate over previous 5 races |
| ConstructorPodiumRate_Last5 | engineered | Team's podium rate over previous 5 races |
| DriverAvgPos_Last5 | engineered | Driver's average finish over previous 5 races |
| ConstructorAvgPos_Last5 | engineered | Team's average finish over previous 5 races |
| DriverAvgGrid_Last5 | engineered | Driver's average grid slot over previous 5 races |
| DriverSeasonPodiums | engineered | Driver's podium count in season-to-date |

### Excluded as leakage
`Position`, `Points`, `Status`, `PositionChange`, `DNF`, `Winner` — all are race **outcomes**, not pre-race info. Including them would let the model "predict" using the answer.

---

## Methodology

1. **Train/test split:** chronological — train on 2000–2022 (9,160 rows), test on 2023–2025 (1,238 rows). This mimics real deployment far better than random split for time-series sports data.
2. **Preprocessing:** median imputation + standard scaling for numeric, mode imputation + one-hot encoding for categorical. Tree models skip scaling.
3. **Class imbalance:** handled via `class_weight='balanced'` (linear/tree) and `scale_pos_weight` (XGBoost) — no resampling.
4. **Models:** Logistic Regression, Decision Tree, k-NN, Random Forest, Gradient Boosting, SVM (RBF), XGBoost.
5. **Evaluation:** Accuracy, Precision, Recall, F1, AUC. Best model selected by F1 (appropriate for imbalanced classification).

---

## Research Questions and Headline Findings

| RQ | Finding |
|---|---|
| **RQ1** Baselines | Logistic Reg AUC = 0.923, Decision Tree F1 = 0.630, k-NN F1 = 0.588 — all viable. |
| **RQ2** Best model | **XGBoost F1 = 0.655**, AUC = 0.927. Random Forest close second (F1 = 0.651). |
| **RQ3** Preprocessing | Adding categorical encoding gains ~0.6 F1 points; scaling has no effect on XGBoost. |
| **RQ4** Top features | **Grid position dominates** (importance 0.158 — 6× the next feature). |
| **RQ5** Metric sensitivity | Rankings flip dramatically: SVM is rank #1 by recall but #5 by precision. |
| **RQ6** Robustness | F1 only drops 0.655 → 0.568 with 10% label noise. 5-fold CV std = 0.014. |
| **RQ7** Recommendation | XGBoost for performance; Logistic Regression if interpretability is critical (AUC nearly identical). |

---

## How to Run

### Option 1: Run the script
```bash
git clone <this-repo>
cd f1_podium_prediction
pip install -r requirements.txt
python f1_podium_analysis.py
```
Outputs land in `results/` and `figures/`.

### Option 2: Run the notebook
```bash
jupyter notebook notebooks/F1_Podium_Prediction.ipynb
```
The repository version is **already executed** — outputs are visible without running anything.

---

## Requirements

Python 3.10+ with packages listed in [requirements.txt](requirements.txt). Main dependencies:
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `openpyxl`, `jupyter`

---

## Reproducibility

All randomness is seeded with `RANDOM_STATE = 42`. The chronological split is deterministic.
Re-running the script yields identical metrics to those reported above (within floating-point noise).
