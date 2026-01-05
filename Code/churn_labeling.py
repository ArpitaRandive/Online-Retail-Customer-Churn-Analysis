import pandas as pd
import os

df = pd.read_csv(
    "../data/cleaned/online_retail_clean.csv",
    parse_dates=["InvoiceDate"]
)

print(df.head())

# ---------------------------
# 2. Aggregate to customer level
# ---------------------------
customer_summary = (
    df.groupby("CustomerID")
      .agg(
          first_purchase_date=("InvoiceDate", "min"),
          last_purchase_date=("InvoiceDate", "max")
      )
      .reset_index()
)

# ---------------------------
# 3. Create churn label
# ---------------------------
CHURN_THRESHOLD_DAYS = 90

customer_summary["churn"] = (
    customer_summary["last_purchase_date"]
    < customer_summary["first_purchase_date"] + pd.Timedelta(days=CHURN_THRESHOLD_DAYS)
).astype(int)

# ---------------------------
# 4. Ensure output directory exists
# ---------------------------
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# 5. Save churn labels
# ---------------------------
output_path = os.path.join(OUTPUT_DIR, "customer_churn_labels.csv")
customer_summary.to_csv(output_path, index=False)

# ---------------------------
# 6. Confirmation prints
# ---------------------------
print("✅ Churn labeling complete")
print(f"📁 File saved at: {output_path}")
print("\nChurn distribution:")
print(customer_summary["churn"].value_counts())