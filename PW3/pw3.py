import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("heart.csv")
print(df.shape)
print(df.head())
continuous_cols = df[["age", "chol", "trestbps", "thalach"]]
continuous_cols.hist(bins=15, figsize=(10, 8), edgecolor="black")
plt.tight_layout()
plt.savefig("heart_distributions.png")
plt.show()
from scipy import stats

for col in ["age", "chol", "trestbps", "thalach"]:
  stat, p_value = stats.shapiro(df[col])
  print(f"{col}: p-value = {p_value:.5f}")
  group_disease = df[df["target"] == 1]["thalach"].dropna()
group_healthy = df[df["target"] == 0]["thalach"].dropna()

mean_disease = group_disease.mean()
sem_disease = group_disease.sem()

mean_healthy = group_healthy.mean()
sem_healthy = group_healthy.sem()

print(
    f"Disease group - Mean thalach: {mean_disease:.2f} (±{sem_disease:.2f})"
)
print(
    f"Healthy group - Mean thalach: {mean_healthy:.2f} (±{sem_healthy:.2f})"
)

u_stat, p_val_mann = stats.mannwhitneyu(group_disease, group_healthy)
print(f"Mann-Whitney U test p-value: {p_val_mann:.5e}")
age_col = df["age"].dropna()
thalach_col = df["thalach"].dropna()

corr_coef, p_val_corr = stats.spearmanr(age_col, thalach_col)
print(f"Spearman correlation coefficient: {corr_coef:.3f}")
print(f"Correlation p-value: {p_val_corr:.5e}")

plt.figure(figsize=(6, 5))
plt.scatter(df["age"], df["thalach"], alpha=0.7, edgecolors="k")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate (thalach)")
plt.title("Age vs Max Heart Rate")
plt.savefig("age_vs_thalach.png")
print("\nScatter plot saved as age_vs_thalach.png")
plt.show()

df_chem = pd.read_csv("chemicals_cancer.csv")
print(df_chem.head())

corr_benzene, p_benzene = stats.spearmanr(
    df_chem["benzene"], df_chem["malignancy"]
)
corr_cadmium, p_cadmium = stats.spearmanr(
    df_chem["cadmium"], df_chem["malignancy"]
)

print(
    f"Benzene correlation with malignancy: {corr_benzene:.3f} (p: {p_benzene:.5e})"
)
print(
    f"Cadmium correlation with malignancy: {corr_cadmium:.3f} (p: {p_cadmium:.5e})"
)

df_filtered = df_chem[(df_chem["pollution_index"] > 40) & (df_chem["pollution_index"] < 60)]

corr_benz_filt, p_benz_filt = stats.spearmanr(df_filtered["benzene"], df_filtered["malignancy"])
corr_cad_filt, p_cad_filt = stats.spearmanr(df_filtered["cadmium"], df_filtered["malignancy"])

print(f"Filtered Benzene correlation: {corr_benz_filt:.3f} (p: {p_benz_filt:.5e})")
print(f"Filtered Cadmium correlation: {corr_cad_filt:.3f} (p: {p_cad_filt:.5e})")
target_balance = df["target"].value_counts(normalize=True)
print("Target balance proportions:\n", target_balance)                 