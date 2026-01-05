import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# Paths
# =========================
DATA_PATH = "../data/processed/ml_customer_features_90d.csv"
OUTPUT_DIR = "../visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load data
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["CustomerID", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================
# Train Random Forest
# =========================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42
)
rf.fit(X_train, y_train)

# =========================
# SHAP Explainability
# =========================
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# ---- IMPORTANT FIX ----
# For binary classification, take SHAP values for class 1 (churn)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values[:, :, 1]

# =========================
# SHAP Summary Plot
# =========================
plt.figure()
shap.summary_plot(
    shap_vals,
    X_test,
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary_rf_90d.png", dpi=300)
plt.close()

# =========================
# SHAP Bar Plot (Global Importance)
# =========================
plt.figure()
shap.summary_plot(
    shap_vals,
    X_test,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_feature_importance_rf_90d.png", dpi=300)
plt.close()

print("✅ SHAP explainability plots saved successfully.")
