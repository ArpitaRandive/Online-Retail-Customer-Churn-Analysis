import os
import pandas as pd

# ============================================================
# 1. Load data
# ============================================================
tx = pd.read_csv(
    "../data/cleaned/online_retail_clean.csv",
    parse_dates=["InvoiceDate"]
)

labels = pd.read_csv(
    "../data/processed/customer_churn_labels.csv",
    parse_dates=["first_purchase_date", "last_purchase_date"]
)

print("Transactions shape:", tx.shape)
print("Labels shape:", labels.shape)

# ============================================================
# 2. Merge transactions with churn labels
# ============================================================
df = tx.merge(labels, on="CustomerID", how="inner")
print("Merged dataset shape:", df.shape)

# ============================================================
# 3. Days since first purchase
# ============================================================
df["days_since_first_purchase"] = (
    df["InvoiceDate"] - df["first_purchase_date"]
).dt.days

# Keep only first 30 days of activity
df_30d = df[df["days_since_first_purchase"].between(0, 30)]
print("30-day activity rows:", df_30d.shape)

# ============================================================
# 4. Aggregate early behavior features (CustomerID as INDEX)
# ============================================================
features = (
    df_30d
    .groupby("CustomerID")
    .agg(
        orders_30d=("InvoiceNo", "nunique"),
        total_quantity_30d=("Quantity", "sum"),
        total_spend_30d=("UnitPrice", "sum"),
        avg_order_value_30d=("UnitPrice", "mean"),
        active_days_30d=("InvoiceDate", lambda x: x.dt.date.nunique())
    )
)

print("Feature table shape:", features.shape)

# ============================================================
# 5. Second purchase date (index-based)
# ============================================================
second_purchase = (
    df.sort_values("InvoiceDate")
      .groupby("CustomerID")
      .InvoiceDate
      .nth(1)
      .rename("second_purchase_date")
)

# Join safely using index
features = features.join(second_purchase, how="left")

# ============================================================
# 6. Join churn labels (index-based)
# ============================================================
labels_indexed = (
    labels
    .set_index("CustomerID")[["first_purchase_date", "churn"]]
)

features = features.join(labels_indexed, how="left")

# ============================================================
# 7. Days to second purchase feature
# ============================================================
features["days_to_second_purchase"] = (
    features["second_purchase_date"] - features["first_purchase_date"]
).dt.days

# If customer never returned → strong churn signal
features["days_to_second_purchase"] = features["days_to_second_purchase"].fillna(999)

# ============================================================
# 8. Final ML dataset
# ============================================================
ml_df = (
    features
    .drop(columns=["second_purchase_date", "first_purchase_date"])
    .reset_index()
)

print("\nML dataset preview:")
print(ml_df.head())

print("\nFinal ML dataset shape:", ml_df.shape)

# ============================================================
# 9. Save ML dataset
# ============================================================
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, "ml_customer_features.csv")
ml_df.to_csv(output_path, index=False)

print(f"\n✅ ML feature dataset saved at: {output_path}")
