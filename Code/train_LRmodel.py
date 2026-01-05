import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Load dataset
df = pd.read_csv("../data/processed/ml_customer_features.csv")

print("Dataset shape:", df.shape)
print(df.head())

#Defining features
X = df.drop(columns=["CustomerID", "churn"])
y = df["churn"]

# Training Logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC:", round(roc_auc, 3))

print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test_scaled)))

# Feature Importance
feature_importance = pd.Series(
    model.coef_[0],
    index=X.columns
).sort_values(key=abs, ascending=False)

print("\nFeature importance:")
print(feature_importance)
