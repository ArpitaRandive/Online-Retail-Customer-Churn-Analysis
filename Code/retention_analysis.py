import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Projects\Online-retail-retention-analysis\Data\Processed\retention_cohort_table.csv")
print(df.head())

df["cohort_month"] = pd.to_datetime(df["cohort_month"]).dt.strftime("%Y-%m")
cohort_pivot = df.pivot(
    index="cohort_month",
    columns="months_since_cohort",
    values="retention_percentage"
)

plt.figure(figsize=(14, 8))

sns.heatmap(
    cohort_pivot,
    annot=True,
    fmt=".1f",
    cmap="Blues"
)

plt.title("Customer Retention Cohort Analysis", fontsize=14)
plt.xlabel("Months Since First Purchase")
plt.ylabel("Cohort Month")

plt.tight_layout()
plt.savefig("../visuals/retention_heatmap.png", dpi=150)
plt.close()

print("✅ Retention heatmap saved to visuals/retention_heatmap.png")