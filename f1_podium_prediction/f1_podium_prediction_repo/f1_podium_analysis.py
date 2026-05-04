"""
F1 Podium Prediction — Supervised Learning Analysis
====================================================

Predicting whether a Formula 1 driver will finish on the podium (top 3)
using race entry data (grid position, weather, driver/constructor history)
from the 2000–2025 seasons.

Target variable: Podium (Yes/No) — binary classification.

This script runs the entire experimental pipeline:
- Data loading and cleaning
- Feature engineering (historical form features, no leakage)
- Train/test split (chronological — train on 2000-2022, test on 2023-2025)
- Baseline models (RQ1)
- Advanced/ensemble models (RQ2)
- Preprocessing ablation (RQ3)
- Feature importance / SHAP (RQ4)
- Multi-metric model ranking (RQ5)
- Robustness analysis (RQ6)
- Decision matrix (RQ7)

Outputs: tables (CSV) in results/ and figures (PNG) in figures/.
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")

# Try optional imports — fall back gracefully if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
RANDOM_STATE = 42
DATA_PATH = "data/f1_cleaned.xlsx"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
print("=" * 70)
print("F1 PODIUM PREDICTION — supervised learning pipeline")
print("=" * 70)

df = pd.read_excel(DATA_PATH)
print(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Seasons covered: {df['Season'].min()} – {df['Season'].max()}")
print(f"Podium rate: {(df['Podium'] == 'Yes').mean():.1%}")

# ----------------------------------------------------------------------
# 2. Feature engineering (NO LEAKAGE — only pre-race info)
# ----------------------------------------------------------------------
# Race outcome columns to EXCLUDE (would cause leakage):
#   Position, Points, Status, PositionChange, DNF, Winner
#
# We construct rolling historical features (driver and constructor recent form)
# computed BEFORE the current race, so they only use past data.
# ----------------------------------------------------------------------
print("\n[1/8] Engineering historical form features (no leakage)…")

df = df.sort_values(["Season", "Round"]).reset_index(drop=True)
df["PodiumBinary"] = (df["Podium"] == "Yes").astype(int)

# Rolling driver podium rate over previous 5 races (shifted to exclude current)
df["DriverPodiumRate_Last5"] = (
    df.groupby("DriverID")["PodiumBinary"]
      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
      .fillna(0)
)
# Rolling constructor podium rate over previous 5 races
df["ConstructorPodiumRate_Last5"] = (
    df.groupby("constructorId")["PodiumBinary"]
      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
      .fillna(0)
)
# Driver's average finishing position over previous 5 races
df["DriverAvgPos_Last5"] = (
    df.groupby("DriverID")["Position"]
      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
      .fillna(df["Position"].mean())
)
# Constructor's average finishing position over previous 5 races
df["ConstructorAvgPos_Last5"] = (
    df.groupby("constructorId")["Position"]
      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
      .fillna(df["Position"].mean())
)
# Season-to-date podium count for driver
df["DriverSeasonPodiums"] = (
    df.groupby(["Season", "DriverID"])["PodiumBinary"]
      .transform(lambda s: s.shift(1).cumsum())
      .fillna(0)
)
# Driver's grid position rolling avg (prior form indicator)
df["DriverAvgGrid_Last5"] = (
    df.groupby("DriverID")["Grid"]
      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
      .fillna(df["Grid"].mean())
)

print("  → 6 rolling form features added")

# ----------------------------------------------------------------------
# 3. Define feature set + target
# ----------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Grid", "Rain", "Round", "Season",
    "DriverPodiumRate_Last5", "ConstructorPodiumRate_Last5",
    "DriverAvgPos_Last5", "ConstructorAvgPos_Last5",
    "DriverSeasonPodiums", "DriverAvgGrid_Last5",
]
CATEGORICAL_FEATURES = ["ConstructorName", "DriverNationality", "country"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "PodiumBinary"

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

# ----------------------------------------------------------------------
# 4. Chronological train/test split
#    Train: 2000–2022 | Test: 2023–2025
#    (More realistic than random split for time-series sports data)
# ----------------------------------------------------------------------
print("\n[2/8] Chronological train/test split (train: 2000-2022, test: 2023-2025)…")

train_mask = df["Season"] <= 2022
X_train = X.loc[train_mask].reset_index(drop=True)
X_test = X.loc[~train_mask].reset_index(drop=True)
y_train = y.loc[train_mask].reset_index(drop=True)
y_test = y.loc[~train_mask].reset_index(drop=True)

print(f"  Train: {len(X_train):,} rows | podium rate: {y_train.mean():.1%}")
print(f"  Test:  {len(X_test):,} rows | podium rate: {y_test.mean():.1%}")

# ----------------------------------------------------------------------
# 5. Preprocessor
# ----------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUMERIC_FEATURES),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL_FEATURES),
])

# Tree-based version (no scaling needed — keeps numeric raw)
preprocessor_tree = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL_FEATURES),
])


def evaluate(model, X_te, y_te):
    """Run a fitted model on test set and return all classification metrics."""
    y_pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_te)
    else:
        y_score = y_pred
    return {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred, zero_division=0),
        "F1": f1_score(y_te, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_te, y_score),
    }


# ======================================================================
# RQ1 — BASELINE PERFORMANCE
# ======================================================================
print("\n[3/8] RQ1 — Baseline models (Logistic / Decision Tree / k-NN)…")

baseline_models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=RANDOM_STATE)),
    ]),
    "Decision Tree": Pipeline([
        ("prep", preprocessor_tree),
        ("clf", DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                        random_state=RANDOM_STATE)),
    ]),
    "k-NN": Pipeline([
        ("prep", preprocessor),
        ("clf", KNeighborsClassifier(n_neighbors=15)),
    ]),
}

baseline_results = {}
for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    baseline_results[name] = evaluate(model, X_test, y_test)
    print(f"  {name:<25s} → Acc={baseline_results[name]['Accuracy']:.3f} "
          f"F1={baseline_results[name]['F1']:.3f} AUC={baseline_results[name]['AUC']:.3f}")

table_I = pd.DataFrame(baseline_results).T.round(3)
table_I.index.name = "Model"
table_I.to_csv(f"{RESULTS_DIR}/Table_I_baseline_performance.csv")

# Figure 1 — grouped bar chart
fig, ax = plt.subplots(figsize=(9, 5))
table_I[["Accuracy", "Precision", "Recall", "F1", "AUC"]].plot(
    kind="bar", ax=ax, colormap="viridis", edgecolor="black", width=0.8)
ax.set_title("Figure 1. Baseline Model Performance on F1 Podium Prediction")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.0)
ax.legend(loc="lower right", ncol=5, fontsize=9)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/Figure_1_baseline_performance.png")
plt.close()


# ======================================================================
# RQ2 — MODEL COMPARISON (advanced + ensembles)
# ======================================================================
print("\n[4/8] RQ2 — Model comparison (advanced + ensembles)…")

advanced_models = {
    "Logistic Regression": baseline_models["Logistic Regression"],
    "Random Forest": Pipeline([
        ("prep", preprocessor_tree),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=15,
                                        class_weight="balanced",
                                        n_jobs=-1, random_state=RANDOM_STATE)),
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", preprocessor_tree),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            random_state=RANDOM_STATE)),
    ]),
    "SVM (RBF)": Pipeline([
        ("prep", preprocessor),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                    random_state=RANDOM_STATE)),
    ]),
}
if HAS_XGB:
    advanced_models["XGBoost"] = Pipeline([
        ("prep", preprocessor_tree),
        ("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                               scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                               eval_metric="logloss", n_jobs=-1,
                               random_state=RANDOM_STATE)),
    ])

advanced_results = {}
for name, model in advanced_models.items():
    if name in baseline_results:
        # already fit
        advanced_results[name] = baseline_results[name]
        continue
    model.fit(X_train, y_train)
    advanced_results[name] = evaluate(model, X_test, y_test)
    print(f"  {name:<25s} → Acc={advanced_results[name]['Accuracy']:.3f} "
          f"F1={advanced_results[name]['F1']:.3f} AUC={advanced_results[name]['AUC']:.3f}")

table_II = pd.DataFrame(advanced_results).T.round(3)
table_II.index.name = "Model"
table_II.to_csv(f"{RESULTS_DIR}/Table_II_model_comparison.csv")

# Identify best model (by F1 — appropriate for imbalanced classification)
BEST_MODEL_NAME = table_II["F1"].idxmax()
BEST_MODEL = advanced_models[BEST_MODEL_NAME]
print(f"\n  ★ Best model by F1: {BEST_MODEL_NAME}")

# Figure 2 — horizontal bar chart of F1 + AUC
fig, ax = plt.subplots(figsize=(9, 5))
plot_df = table_II[["F1", "AUC"]].sort_values("F1")
plot_df.plot(kind="barh", ax=ax, colormap="plasma", edgecolor="black")
ax.set_title("Figure 2. Overall Model Ranking — F1 and AUC")
ax.set_xlabel("Score")
ax.set_xlim(0, 1.0)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/Figure_2_model_ranking.png")
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in advanced_models.items():
    if hasattr(model.named_steps["clf"], "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    ax.plot(fpr, tpr, label=f"{name} (AUC={advanced_results[name]['AUC']:.3f})", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — F1 Podium Prediction")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/Figure_2b_roc_curves.png")
plt.close()


# ======================================================================
# RQ3 — EFFECT OF PREPROCESSING (ablation on best model)
# ======================================================================
print(f"\n[5/8] RQ3 — Preprocessing ablation on {BEST_MODEL_NAME}…")

# Define preprocessing variants
def make_variant(strategy):
    """Build a pipeline with the same classifier but different preprocessing."""
    clf = type(BEST_MODEL.named_steps["clf"])(**BEST_MODEL.named_steps["clf"].get_params())

    if strategy == "raw":
        # Numeric only, no scaling, no encoding (drop categoricals)
        prep = ColumnTransformer([
            ("num", "passthrough", NUMERIC_FEATURES),
        ], remainder="drop")
    elif strategy == "imputation":
        # Numeric only with imputation, no scaling/encoding
        prep = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ], remainder="drop")
    elif strategy == "scaling_encoding":
        # Numeric scaled + categorical encoded but no imputation tuning
        prep = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ])
    else:  # full
        prep = preprocessor_tree if BEST_MODEL_NAME != "Logistic Regression" else preprocessor

    return Pipeline([("prep", prep), ("clf", clf)])


prep_variants = {
    "Raw data (numeric only)": make_variant("raw"),
    "Imputation": make_variant("imputation"),
    "Scaling + Encoding": make_variant("scaling_encoding"),
    "Full pipeline": make_variant("full"),
}

prep_results = {}
for name, model in prep_variants.items():
    try:
        model.fit(X_train, y_train)
        prep_results[name] = evaluate(model, X_test, y_test)
        print(f"  {name:<28s} → Acc={prep_results[name]['Accuracy']:.3f} "
              f"F1={prep_results[name]['F1']:.3f} AUC={prep_results[name]['AUC']:.3f}")
    except Exception as e:
        print(f"  {name}: failed ({e})")
        prep_results[name] = {k: np.nan for k in ["Accuracy", "Precision", "Recall", "F1", "AUC"]}

table_III = pd.DataFrame(prep_results).T.round(3)
table_III.index.name = "Preprocessing Strategy"
table_III.to_csv(f"{RESULTS_DIR}/Table_III_preprocessing_ablation.csv")

# Figure 3
fig, ax = plt.subplots(figsize=(9, 5))
table_III[["Accuracy", "F1", "AUC"]].plot(kind="bar", ax=ax,
                                            colormap="cividis", edgecolor="black", width=0.8)
ax.set_title(f"Figure 3. Preprocessing Ablation ({BEST_MODEL_NAME})")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/Figure_3_preprocessing_ablation.png")
plt.close()


# ======================================================================
# RQ4 — FEATURE IMPORTANCE
# ======================================================================
print(f"\n[6/8] RQ4 — Feature importance from {BEST_MODEL_NAME}…")

# Extract feature names after preprocessing
fitted_prep = BEST_MODEL.named_steps["prep"]
feat_names = []
for name, trans, cols in fitted_prep.transformers_:
    if name == "num":
        feat_names.extend(cols)
    elif name == "cat":
        if hasattr(trans, "named_steps"):
            ohe = trans.named_steps["ohe"]
        else:
            ohe = trans
        feat_names.extend(ohe.get_feature_names_out(cols).tolist())

clf = BEST_MODEL.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
elif hasattr(clf, "coef_"):
    importances = np.abs(clf.coef_[0])
else:
    importances = np.zeros(len(feat_names))

importance_df = (pd.DataFrame({"Feature": feat_names, "Importance": importances})
                 .sort_values("Importance", ascending=False))
top10 = importance_df.head(10).copy()
top10["Direction"] = ["Positive" if v > 0 else "Negative" for v in top10["Importance"]]
top10.insert(0, "Rank", range(1, len(top10) + 1))


def interpret(feat):
    f = feat.lower()
    if "grid" in f and "driver" not in f:
        return "Lower grid number (front of grid) increases podium odds"
    if "driveravggrid" in f.lower():
        return "Driver's recent qualifying form"
    if "driverpodiumrate" in f.lower():
        return "Recent driver form on podium"
    if "constructorpodiumrate" in f.lower():
        return "Team's recent podium track record"
    if "driveravgpos" in f.lower():
        return "Driver's recent finishing strength"
    if "constructoravgpos" in f.lower():
        return "Team's recent finishing strength"
    if "driverseasonpodiums" in f.lower():
        return "Driver's podium count this season"
    if "constructorname" in f.lower():
        return "Specific team identity (top teams dominate)"
    if "rain" in f.lower():
        return "Wet conditions can shuffle the order"
    if "drivernationality" in f.lower() or "country" in f.lower():
        return "Geographic/regional effect"
    if "season" in f.lower():
        return "Era effect across years"
    if "round" in f.lower():
        return "Calendar position effect"
    return "Contextual predictor"


top10["Domain Interpretation"] = top10["Feature"].apply(interpret)
table_IV = top10[["Rank", "Feature", "Importance", "Direction", "Domain Interpretation"]]
table_IV["Importance"] = table_IV["Importance"].round(4)
table_IV.to_csv(f"{RESULTS_DIR}/Table_IV_feature_importance.csv", index=False)

# Figure 4
fig, ax = plt.subplots(figsize=(9, 6))
top10_plot = top10.iloc[::-1]  # reverse for horizontal bar
ax.barh(top10_plot["Feature"], top10_plot["Importance"], color="steelblue", edgecolor="black")
ax.set_xlabel("Importance Score")
ax.set_title(f"Figure 4. Top 10 Features ({BEST_MODEL_NAME})")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/Figure_4_feature_importance.png")
plt.close()


# ======================================================================
# RQ5 — METRIC SENSITIVITY (model rankings change across metrics)
# ======================================================================
print("\n[7/8] RQ5 — Sensitivity to evaluation metrics…")

ranking_df = table_II.rank(ascending=False, method="min").astype(int)
ranking_df.columns = [f"Rank by {c}" for c in ranking_df.columns]
ranking_df.index.name = "Model"
ranking_df.to_csv(f"{RESULTS_DIR}/Table_V_ranking_sensitivity.csv")
table_V = ranking_df

# Figure 5 — bump chart
fig, ax = plt.subplots(figsize=(10, 5))
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x = np.arange(len(metrics))
colors = plt.cm.tab10(np.linspace(0, 1, len(table_II)))
for i, model in enumerate(table_II.index):
    ranks = [table_II[m].rank(ascending=False).astype(int)[model] for m in metrics]
    ax.plot(x, ranks, marker="o", lw=2, label=model, color=colors[i])
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.invert_yaxis()  # rank 1 at top
ax.set_ylabel("Rank (1 = best)")
ax.set_title("Figure 5. Model Rank Across Evaluation Metrics")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/Figure_5_ranking_sensitivity.png")
plt.close()


# ======================================================================
# RQ6 — ROBUSTNESS ANALYSIS
# ======================================================================
print(f"\n[8/8] RQ6 — Robustness of {BEST_MODEL_NAME}…")

robust_results = {}

# Standard split (already computed)
robust_results["Standard split (chronological)"] = {
    **evaluate(BEST_MODEL, X_test, y_test),
    "Std Dev": 0.000,
}

# 5-fold stratified CV on training data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}
for tr_idx, va_idx in skf.split(X_train, y_train):
    Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
    m = type(BEST_MODEL)(steps=[(n, p) for n, p in BEST_MODEL.steps])
    # Fresh fit
    from sklearn.base import clone
    m = clone(BEST_MODEL)
    m.fit(Xtr, ytr)
    s = evaluate(m, Xva, yva)
    for k in cv_metrics:
        cv_metrics[k].append(s[k])
robust_results["5-fold CV"] = {k: np.mean(v) for k, v in cv_metrics.items()}
robust_results["5-fold CV"]["Std Dev"] = float(np.std(cv_metrics["F1"]))

# 10% noise: corrupt 10% of training labels
rng = np.random.RandomState(RANDOM_STATE)
y_train_noisy = y_train.copy()
flip_idx = rng.choice(len(y_train_noisy), int(0.1 * len(y_train_noisy)), replace=False)
y_train_noisy.iloc[flip_idx] = 1 - y_train_noisy.iloc[flip_idx]
m_noise = clone(BEST_MODEL)
m_noise.fit(X_train, y_train_noisy)
robust_results["10% label noise"] = {**evaluate(m_noise, X_test, y_test), "Std Dev": np.nan}

# 20% missingness: randomly null 20% of numeric values
X_train_miss = X_train.copy()
for col in NUMERIC_FEATURES:
    miss_idx = rng.choice(len(X_train_miss), int(0.2 * len(X_train_miss)), replace=False)
    X_train_miss.loc[miss_idx, col] = np.nan
m_miss = clone(BEST_MODEL)
m_miss.fit(X_train_miss, y_train)
robust_results["20% missingness"] = {**evaluate(m_miss, X_test, y_test), "Std Dev": np.nan}

table_VI = pd.DataFrame(robust_results).T.round(3)
table_VI.index.name = "Scenario"
table_VI.to_csv(f"{RESULTS_DIR}/Table_VI_robustness.csv")

for scen, sc in robust_results.items():
    print(f"  {scen:<35s} → Acc={sc['Accuracy']:.3f} F1={sc['F1']:.3f} AUC={sc['AUC']:.3f}")

# Figure 6 — boxplot of CV scores
fig, ax = plt.subplots(figsize=(9, 5))
cv_df = pd.DataFrame(cv_metrics)
cv_df.boxplot(ax=ax, grid=True)
ax.set_title(f"Figure 6. 5-Fold CV Score Distribution ({BEST_MODEL_NAME})")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/Figure_6_robustness_cv.png")
plt.close()


# ======================================================================
# RQ7 — DECISION MATRIX
# ======================================================================
print("\n[+] RQ7 — Final decision matrix…")

# Build decision matrix programmatically based on actual results
def grade(value, thresholds=(0.6, 0.75, 0.85)):
    """Convert a score to qualitative label."""
    if value >= thresholds[2]:
        return "Very High"
    if value >= thresholds[1]:
        return "High"
    if value >= thresholds[0]:
        return "Medium"
    return "Low"


def interpretability_grade(name):
    return {
        "Logistic Regression": "High",
        "Decision Tree": "High",
        "k-NN": "Low",
        "Random Forest": "Medium",
        "Gradient Boosting": "Medium-Low",
        "XGBoost": "Medium-Low",
        "SVM (RBF)": "Low",
    }.get(name, "Medium")


def cost_grade(name):
    return {
        "Logistic Regression": "Low",
        "Decision Tree": "Low",
        "k-NN": "Medium",
        "Random Forest": "Medium",
        "Gradient Boosting": "High",
        "XGBoost": "High",
        "SVM (RBF)": "High",
    }.get(name, "Medium")


def deploy_grade(name):
    return {
        "Logistic Regression": "Very High",
        "Decision Tree": "High",
        "k-NN": "Medium",
        "Random Forest": "High",
        "Gradient Boosting": "Medium",
        "XGBoost": "Medium",
        "SVM (RBF)": "Low",
    }.get(name, "Medium")


criteria = []
for model in table_II.index:
    criteria.append({
        "Model": model,
        "Predictive Performance (F1)": grade(table_II.loc[model, "F1"]),
        "Predictive Performance (AUC)": grade(table_II.loc[model, "AUC"]),
        "Interpretability": interpretability_grade(model),
        "Computational Cost": cost_grade(model),
        "Deployment Suitability": deploy_grade(model),
    })
table_VII = pd.DataFrame(criteria).set_index("Model")
table_VII.loc["★ Selected Model", :] = [
    grade(table_II.loc[BEST_MODEL_NAME, "F1"]),
    grade(table_II.loc[BEST_MODEL_NAME, "AUC"]),
    interpretability_grade(BEST_MODEL_NAME),
    cost_grade(BEST_MODEL_NAME),
    deploy_grade(BEST_MODEL_NAME),
]
table_VII.to_csv(f"{RESULTS_DIR}/Table_VII_decision_matrix.csv")

# Figure 7 — radar chart of best model on quantitative metrics
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
angles += angles[:1]
for model in table_II.index:
    vals = [table_II.loc[model, m] for m in metric_labels]
    vals += vals[:1]
    ax.plot(angles, vals, lw=2, label=model)
    ax.fill(angles, vals, alpha=0.08)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.0)
ax.set_title("Figure 7. Multi-Metric Profile per Model", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/Figure_7_decision_radar.png")
plt.close()


# ======================================================================
# Confusion matrix for the best model (extra figure for the report)
# ======================================================================
y_pred_best = BEST_MODEL.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Podium", "Podium"],
            yticklabels=["No Podium", "Podium"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {BEST_MODEL_NAME} (Test Set)")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/Figure_extra_confusion_matrix.png")
plt.close()


# ----------------------------------------------------------------------
# Wrap up — save a summary
# ----------------------------------------------------------------------
summary = {
    "best_model": BEST_MODEL_NAME,
    "test_metrics": {k: float(v) for k, v in advanced_results[BEST_MODEL_NAME].items()},
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "train_seasons": "2000-2022",
    "test_seasons": "2023-2025",
    "n_features": int(len(ALL_FEATURES)),
    "podium_rate_train": float(y_train.mean()),
    "podium_rate_test": float(y_test.mean()),
}
with open(f"{RESULTS_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print(f"DONE. All tables saved to {RESULTS_DIR}/, figures to {FIGURES_DIR}/")
print(f"Best model: {BEST_MODEL_NAME}")
print(f"Test F1={advanced_results[BEST_MODEL_NAME]['F1']:.3f}, "
      f"AUC={advanced_results[BEST_MODEL_NAME]['AUC']:.3f}")
print("=" * 70)
