import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load ML dataset
df = pd.read_csv("../data/processed/ml_customer_features_90d.csv")

X = df.drop(columns=["CustomerID", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Train Random forest Model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Evaluation
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)

print("Random Forest ROC-AUC:", round(rf_auc, 3))
print("\nClassification Report (RF):")
print(classification_report(y_test, rf.predict(X_test)))

# Feature importance
rf_importance = (
    pd.Series(rf.feature_importances_, index=X.columns)
      .sort_values(ascending=False)
)

print("\nRandom Forest Feature Importance:")
print(rf_importance)

## XGBoost 
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

#Evaluation
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)

print("XGBoost ROC-AUC:", round(xgb_auc, 3))
print("\nClassification Report (XGB):")
print(classification_report(y_test, xgb.predict(X_test)))

#Feature Importance
xgb_importance = (
    pd.Series(xgb.feature_importances_, index=X.columns)
      .sort_values(ascending=False)
)

print("\nXGBoost Feature Importance:")
print(xgb_importance)
