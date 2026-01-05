import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# ------------------------------------------------------------
# Model performance summary 
# ------------------------------------------------------------
data = [
    # window, model, roc_auc
    ("14 days", "Random Forest", 0.567),
    ("14 days", "XGBoost", 0.580),

    ("30 days", "Logistic Regression", 0.543),
    ("30 days", "Random Forest", 0.583),
    ("30 days", "XGBoost", 0.576),

    ("60 days", "Random Forest", 0.627),
    ("60 days", "XGBoost", 0.608),

    ("90 days", "Random Forest", 0.682),
    ("90 days", "XGBoost", 0.661),
]

df = pd.DataFrame(data, columns=["Window", "Model", "ROC_AUC"])
pivot = df.pivot(index="Window", columns="Model", values="ROC_AUC")
plt.figure(figsize=(10, 6))
pivot.plot(marker="o")

plt.ylabel("ROC-AUC")
plt.xlabel("Observation Window")
plt.title("Churn Model Performance Across Observation Windows")
plt.ylim(0.5, 0.72)

plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("../visuals/model_performance_by_window.png", dpi=150)
plt.show()


# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------
df = pd.read_csv("../data/processed/ml_customer_features_90d.csv")

X = df.drop(columns=["CustomerID", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss", random_state=42
    ),
}

plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
plt.title("ROC Curves – Churn Prediction (90-Day Window)", fontsize=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("../visuals/roc_curves_90d.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 60d vs 90d AUC
# ------------------------------------------------------------
data = {
    "Window": ["60 days", "60 days", "90 days", "90 days"],
    "Model": ["Random Forest", "XGBoost", "Random Forest", "XGBoost"],
    "ROC_AUC": [0.627, 0.608, 0.682, 0.661],
}

df = pd.DataFrame(data)
pivot = df.pivot(index="Window", columns="Model", values="ROC_AUC")
plt.figure(figsize=(7, 5))

pivot.plot(
    kind="bar",
    width=0.65,
    edgecolor="black"
)

plt.ylabel("ROC-AUC")
plt.xlabel("Observation Window")
plt.title("Churn Model Performance: 60 vs 90 Days")

plt.ylim(0.55, 0.72)
plt.grid(axis="y", linestyle="--", alpha=0.6)
for container in plt.gca().containers:
    plt.bar_label(container, fmt="%.3f", padding=3)

plt.legend(title="Model")
plt.tight_layout()
plt.savefig("../visuals/auc_60_vs_90_comparison.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------
df = pd.read_csv(
    "../data/processed/ml_customer_features_90d.csv"
)

X = df.drop(columns=["CustomerID", "churn"])
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
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
plt.title("Confusion Matrix – Random Forest (90-Day Window)")

plt.tight_layout()
plt.savefig("../visuals/confusion_matrix_rf_90d.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------
rf_importance = pd.Series({
    "active_days_w": 0.3047,
    "orders_w": 0.2358,
    "total_quantity_w": 0.1890,
    "total_spend_w": 0.1399,
    "avg_order_value_w": 0.1195,
    "days_to_second_purchase": 0.0111
})

xgb_importance = pd.Series({
    "active_days_w": 0.4043,
    "orders_w": 0.1696,
    "days_to_second_purchase": 0.1303,
    "avg_order_value_w": 0.0992,
    "total_quantity_w": 0.0991,
    "total_spend_w": 0.0975
})

df = pd.DataFrame({
    "Random Forest": rf_importance,
    "XGBoost": xgb_importance
}).fillna(0)

df = df.sort_values("Random Forest")

df.plot(kind="barh", figsize=(10, 6))
plt.title("Feature Importance Comparison (90-Day Window)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("../visuals/feature_importance_comparison.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Lift Curve
# ------------------------------------------------------------
df = pd.read_csv(
    "../data/processed/ml_customer_features_90d.csv"
)

X = df.drop(columns=["CustomerID", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)
y_proba = rf.predict_proba(X_test)[:, 1]
lift_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_score": y_proba
})

lift_df = lift_df.sort_values("y_score", ascending=False)
lift_df["decile"] = pd.qcut(lift_df.index, 10, labels=False)
lift_table = lift_df.groupby("decile").agg(
    customers=("y_true", "count"),
    churners=("y_true", "sum")
).reset_index()

lift_table["cum_customers"] = lift_table["customers"].cumsum()
lift_table["cum_churners"] = lift_table["churners"].cumsum()

total_churners = lift_table["churners"].sum()

lift_table["cum_churn_rate"] = lift_table["cum_churners"] / total_churners
lift_table["population_pct"] = lift_table["cum_customers"] / lift_table["customers"].sum()


plt.figure(figsize=(8, 6))

plt.plot(
    lift_table["population_pct"],
    lift_table["cum_churn_rate"],
    marker="o",
    label="Random Forest Model"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Targeting"
)

plt.xlabel("Percentage of Customers Targeted")
plt.ylabel("Percentage of Churners Captured")
plt.title("Lift Curve – Churn Prediction (90-Day Window)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../visuals/lift_curve_rf_90d.png", dpi=300)
plt.show()


print("✅ All visuals generated successfully")