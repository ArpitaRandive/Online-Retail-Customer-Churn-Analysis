import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# =========================
# Paths
# =========================
DATA_PATH = "../data/processed/ml_customer_features_90d.csv"
OUTPUT_DIR = "../visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Business Costs (EDITABLE)
# =========================
COST_FN = 10   # Missing a churner
COST_FP = 2    # Unnecessary retention offer

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

y_prob = rf.predict_proba(X_test)[:, 1]

# =========================
# Cost-Based Threshold Search
# =========================
thresholds = np.linspace(0.05, 0.95, 200)
total_costs = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cost = COST_FN * fn + COST_FP * fp
    total_costs.append(cost)

total_costs = np.array(total_costs)
best_idx = np.argmin(total_costs)
best_threshold = thresholds[best_idx]

print("Cost-Optimized Threshold:", round(best_threshold, 3))
print("Minimum Total Cost:", total_costs[best_idx])

# =========================
# Plot Cost Curve
# =========================
plt.figure(figsize=(10, 6))
plt.plot(thresholds, total_costs, linewidth=2)
plt.axvline(
    best_threshold,
    color="red",
    linestyle="--",
    label=f"Optimal Threshold = {best_threshold:.2f}"
)

plt.xlabel("Probability Threshold")
plt.ylabel("Total Business Cost")
plt.title("Cost-Based Threshold Optimization – Random Forest (90-Day Window)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/cost_based_threshold_rf_90d.png", dpi=300)
plt.show()
