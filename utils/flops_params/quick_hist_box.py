import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, kendalltau
import numpy as np


df = pd.read_csv('file2.csv')

print(df.shape)
df.dropna(subset=['FLOPs'], inplace=True)

# Compute paired differences
diff = df['FLOPs'] - df['flopsFast']

n = len(diff)
print(f"Number of paired samples: {n}")

# ---------- 1. Normality test (Shapiro-Wilk) ----------
# Note: For large n (>500), this test is overly sensitive
shapiro_stat, shapiro_p = stats.shapiro(diff.sample(500, random_state=0) if n > 500 else diff)

print("\nShapiro-Wilk normality test (on differences):")
print(f"  statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4e}")

# ---------- 2. Skewness and kurtosis ----------
skew = stats.skew(diff)
kurt = stats.kurtosis(diff)

print("\nShape statistics of differences:")
print(f"  skewness  = {skew:.4f}")
print(f"  kurtosis  = {kurt:.4f}  (0 = normal)")

# ---------- 3. Visual diagnostics ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Histogram
axes[0].hist(diff, bins=30)
axes[0].set_title("Histogram of Differences")

# Q-Q plot
stats.probplot(diff, plot=axes[1])
axes[1].set_title("Q–Q Plot of Differences")

plt.tight_layout()
plt.show()

# ---------- 4. Outlier check (IQR rule) ----------
q1, q3 = np.percentile(diff, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outlier_frac = ((diff < lower) | (diff > upper)).mean()

print("\nOutlier analysis (IQR method):")
print(f"  Fraction of outliers: {outlier_frac:.3%}")

# ---------- 5. Recommendation ----------
print("\nRecommendation:")
if shapiro_p > 0.05 and abs(skew) < 1 and outlier_frac < 0.05:
    print("  ✔ Paired t-test is reasonable.")
else:
    print("  ⚠ Assumptions questionable — consider Wilcoxon signed-rank test.")
#===================================================

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Histograms
axes[0, 0].hist(df['FLOPs'])
axes[0, 0].set_title('Histogram of FLOPs')

axes[0, 1].hist(df['flopsFast'])
axes[0, 1].set_title('Histogram of flopsFast')

# Boxplots
axes[1, 0].boxplot(df['FLOPs'])
axes[1, 0].set_title('Boxplot of FLOPs')

axes[1, 1].boxplot(df['flopsFast'])
axes[1, 1].set_title('Boxplot of flopsFast')

# Improve layout
plt.tight_layout()
plt.show()

