# 🏎️ Predicting Formula 1 Podium Finishes Using Supervised Machine Learning

**DS120B — Machine Learning | Summer Semester 2026**

---

## 📋 Project Overview

This project applies supervised machine learning to predict whether a Formula 1 driver will finish on the **podium (top 3)** in a given race, using race data from the 2000–2025 seasons.

| Field | Details |
|-------|---------|
| **Task** | Binary Classification |
| **Target Variable** | `podium` (1 = top 3 finish, 0 = non-podium) |
| **Dataset** | Formula 1 Races Data from 2000–2025 |
| **Rows × Columns** | 10,398 × 20 |
| **Models** | Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM |

---

## 📊 Dataset

**Source:** [Formula 1 Races Data from 2000–2025 — Kaggle](https://www.kaggle.com/datasets/syedzubairahmed0022/formula-1-races-data-from-2000-2025)

### Download Instructions

1. Visit the Kaggle dataset link above
2. Download the CSV file
3. Place it in the root directory of this repository (same folder as the notebook)

> The notebook auto-detects any CSV file named with `formula`, `f1`, or `race` in its filename. If the file has a different name, rename it or update the path in Cell 2.

---

## 🗂️ Repository Structure

```
📦 f1-podium-prediction/
├── 📓 f1_podium_prediction.ipynb   # Main Jupyter notebook (fully executed)
├── 📄 ML_Assignment1_Proposal.docx # Assignment proposal document
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # This file
└── 📊 figures/                      # Generated output figures
    ├── fig1_class_distribution.png
    ├── fig2_grid_vs_podium.png
    ├── fig3_correlation_heatmap.png
    ├── fig4_baseline_performance.png
    ├── fig5_model_comparison.png
    ├── fig6_preprocessing_ablation.png
    ├── fig7_feature_importance.png
    ├── fig8_roc_curves.png
    ├── fig9_confusion_matrix.png
    ├── fig10_robustness_boxplot.png
    ├── fig11_bump_chart.png
    └── fig12_radar_chart.png
```

---

## 🚀 How to Run

### Option 1: Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/f1-podium-prediction.git
cd f1-podium-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and place CSV in this directory

# 5. Launch Jupyter
jupyter notebook f1_podium_prediction.ipynb
# OR
jupyter lab f1_podium_prediction.ipynb
```

### Option 2: Google Colab

1. Upload `f1_podium_prediction.ipynb` to Google Colab
2. Upload the dataset CSV to Colab's file system
3. Run all cells (Runtime → Run all)

---

## 🔬 Research Questions

| # | Research Question |
|---|------------------|
| RQ1 | How effectively can **baseline models** predict F1 podium finishes? |
| RQ2 | Which model achieves the **best predictive performance**? |
| RQ3 | How do **preprocessing strategies** affect classification performance? |
| RQ4 | Which **features** are most important for podium prediction? |
| RQ5 | How do model rankings change across **different evaluation metrics**? |
| RQ6 | How **robust** is the best model under cross-validation and noise? |
| RQ7 | Which model is most **practically useful** for real-world deployment? |

---

## 🤖 Models Used

| Model | Library | Notes |
|-------|---------|-------|
| Logistic Regression | scikit-learn | Linear baseline |
| Decision Tree | scikit-learn | Non-linear baseline |
| k-Nearest Neighbours | scikit-learn | Instance-based baseline |
| Random Forest | scikit-learn | Ensemble — primary model |
| XGBoost | xgboost | Gradient boosting — best performance |
| SVM | scikit-learn | Kernel-based classifier |

---

## 📈 Evaluation Metrics

- **Accuracy** — Overall correctness
- **Precision** — Quality of podium predictions
- **Recall** — Coverage of actual podium finishes
- **F1-Score** — Primary metric (handles class imbalance)
- **AUC-ROC** — Ranking quality across thresholds
- **Confusion Matrix** — Detailed error breakdown

---

## 📦 Requirements

See `requirements.txt`. Key dependencies:

```
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 📄 License

This project is submitted as part of the DS120B Machine Learning course, SS26.

---

*Dataset provided by [Syed Zubair Ahmed](https://www.kaggle.com/syedzubairahmed0022) on Kaggle.*
