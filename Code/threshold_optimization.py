import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix
import seaborn as sns

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
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================
# Train Random Forest (best model)
# =========================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42
)
rf.fit(X_train, y_train)

# =========================
# Predicted probabilities
# =========================
y_prob = rf.predict_proba(X_test)[:, 1]

# =========================
# Threshold Optimization
# =========================
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# sklearn quirk: last precision/recall has no threshold
precision = precision[:-1]
recall = recall[:-1]

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("\nOptimal Threshold (F1):", round(best_threshold, 3))
print("Precision:", round(precision[best_idx], 3))
print("Recall:", round(recall[best_idx], 3))
print("F1-score:", round(f1_scores[best_idx], 3))

# =========================
# Plot Threshold vs Metrics
# =========================
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision, label="Precision")
plt.plot(thresholds, recall, label="Recall")
plt.plot(thresholds, f1_scores, label="F1-score", linewidth=2)

plt.axvline(
    best_threshold,
    color="red",
    linestyle="--",
    label=f"Optimal Threshold = {best_threshold:.2f}"
)

plt.xlabel("Probability Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization – Random Forest (90-Day Window)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/threshold_optimization_rf_90d.png", dpi=300)
plt.show()

# ==========================================================
# ✅ FINAL STEP: Confusion Matrix @ Optimized Threshold
# ==========================================================

# Apply optimized threshold
y_pred_opt = (y_prob >= best_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_opt)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Churned", "Churned"],
    yticklabels=["Not Churned", "Churned"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Optimized Threshold (RF, 90-Day Window)")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_rf_90d_optimized.png", dpi=300)
plt.show()
