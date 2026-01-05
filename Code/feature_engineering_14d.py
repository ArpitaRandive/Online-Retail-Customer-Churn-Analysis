import os
import pandas as pd

# ============================================================
# PARAMETERS
# ============================================================
WINDOW_DAYS = 14
OUTPUT_NAME = "ml_customer_features_14d.csv"

# ============================================================
# Load data
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
# Merge transactions with labels
# ============================================================
df = tx.merge(labels, on="CustomerID", how="inner")

# ============================================================
# Days since first purchase
# ============================================================
df["days_since_first_purchase"] = (
    df["InvoiceDate"] - df["first_purchase_date"]
).dt.days

df_window = df[df["days_since_first_purchase"].between(0, WINDOW_DAYS)]
print(f"{WINDOW_DAYS}-day activity rows:", df_window.shape)

# ============================================================
# Aggregate early behavior features (index-based)
# ============================================================
features = (
    df_window
    .groupby("CustomerID")
    .agg(
        orders_w=("InvoiceNo", "nunique"),
        total_quantity_w=("Quantity", "sum"),
        total_spend_w=("UnitPrice", "sum"),
        avg_order_value_w=("UnitPrice", "mean"),
        active_days_w=("InvoiceDate", lambda x: x.dt.date.nunique())
    )
)

print("Feature table shape:", features.shape)

# ============================================================
# Second purchase date
# ============================================================
second_purchase = (
    df.sort_values("InvoiceDate")
      .groupby("CustomerID")
      .InvoiceDate
      .nth(1)
      .rename("second_purchase_date")
)

features = features.join(second_purchase, how="left")

# ============================================================
# Join churn labels
# ============================================================
labels_indexed = (
    labels
    .set_index("CustomerID")[["first_purchase_date", "churn"]]
)

features = features.join(labels_indexed, how="left")

# ============================================================
# Days to second purchase
# ============================================================
features["days_to_second_purchase"] = (
    features["second_purchase_date"] - features["first_purchase_date"]
).dt.days

features["days_to_second_purchase"] = features["days_to_second_purchase"].fillna(999)

# ============================================================
# Final ML dataset
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
# Save
# ============================================================
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
ml_df.to_csv(output_path, index=False)

print(f"\n✅ 14-day ML dataset saved at: {output_path}")
