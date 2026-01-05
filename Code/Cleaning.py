import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\Projects\Online-retail-retention-analysis\Data\Raw\online_retail.csv")
print(df.head())

# Removing redundant data
df = df[df["CustomerID"].notna()]
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["UnitPrice"] > 0]


print("\nFinal cleaned dataset info:")
print(df.info())
print("\nFinal shape:", df.shape)

# Remove invalid quantities (returns)
df = df[df["Quantity"] > 0]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M")
df["CustomerID"] = df["CustomerID"].astype(int)

print("\nFinal dataset info:")
print(df.info())

print("\nFinal shape:", df.shape)

print("\nCheck invalid values:")
print("Quantity <= 0:", (df["Quantity"] <= 0).sum())
print("UnitPrice <= 0:", (df["UnitPrice"] <= 0).sum())
print("Missing CustomerID:", df["CustomerID"].isna().sum())

# Save updated file
df.to_csv("../data/cleaned/online_retail_clean.csv", index=False)
print("✅ Cleaned dataset saved successfully")
