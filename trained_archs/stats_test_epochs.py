import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def calculate_spearman_correlation(excel_file, reference_epochs=100):
    """
    Reads an Excel file with architecture training data and calculates Spearman rank correlation
    between accuracy rankings at different epoch counts and a reference epoch count.
    
    Parameters:
    - excel_file (str): Path to the Excel file.
    - reference_epochs (int): The epoch count to use as the "fully trained" reference (default: 100).
    
    Returns:
    - dict: Spearman correlations for each epoch count compared to the reference.
    - int: Recommended epoch count for the proxy task.
    """
    
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(excel_file)
    
    # Check for duplicate ID-Epoch combinations
    if df.duplicated(subset=['ID', 'Epochs']).any():
        print("Warning: Duplicate (ID, Epochs) pairs found! Aggregating using mean.")
        df = df.groupby(['ID', 'Epochs'], as_index=False).mean()

    # Extract unique epoch counts and architecture IDs
    epoch_counts = sorted(df['Epochs'].unique())

    # Ensure the reference epoch count exists in the data
    if reference_epochs not in epoch_counts:
        raise ValueError(f"Reference epoch count {reference_epochs} not found in the data.")
    
    # Pivot the data to get accuracies for each ID at each epoch count
    pivot_df = df.pivot(index='ID', columns='Epochs', values='Accuracy')

    # Drop rows where any epoch's accuracy is missing
    pivot_df = pivot_df.dropna()

    # Get the reference accuracies (at 100 epochs, or specified reference)
    reference_accuracies = pivot_df[reference_epochs]
    reference_ranks = reference_accuracies.rank(ascending=False)  # Higher accuracy = lower rank number
    
    # Dictionary to store Spearman correlations
    correlations = {}

    # Calculate Spearman correlation for each non-reference epoch count
    for epochs in epoch_counts:
        if epochs == reference_epochs:
            continue
        accuracies = pivot_df[epochs]
        ranks = accuracies.rank(ascending=False)
        
        # Compute Spearman correlation
        correlation, _ = spearmanr(ranks, reference_ranks)
        correlations[epochs] = correlation

    # Print results
    print(f"Spearman Rank Correlations (vs. {reference_epochs} epochs):")
    for epochs, corr in correlations.items():
        print(f"{epochs} epochs: {corr:.4f}")

    # Recommend the epoch count with the highest correlation
    recommended_epochs = max(correlations, key=correlations.get)
    print(f"\nRecommended proxy epoch count: {recommended_epochs} (Correlation: {correlations[recommended_epochs]:.4f})")
    
    return correlations, recommended_epochs


# Load the Excel file
excel_file_path = "trained_archs_epochs30"
df = pd.read_excel(excel_file_path + '.xlsx', sheet_name=excel_file_path)

# Group data by Epochs
epoch_groups = df.groupby("Epochs")["Accuracy"]

# Perform ANOVA test
anova_stat, anova_p = stats.f_oneway(*[group for _, group in epoch_groups])
print(f"ANOVA Test: F-statistic = {anova_stat:.3f}, p-value = {anova_p:.3f}")
if anova_p < 0.05:
    print("Conclusion: There is a significant difference in accuracy across epoch groups.")
else:
    print("Conclusion: No significant difference in accuracy across epoch groups.")
print()

# Perform Tukey's HSD test if ANOVA is significant
if anova_p < 0.05:
    tukey = pairwise_tukeyhsd(df["Accuracy"], df["Epochs"], alpha=0.05)
    print("\nTukey's HSD Test Results:")
    print(tukey.summary())
print()

# Compute correlation between 5 and 100 epochs
short_epochs = df[df["Epochs"] == 5]["Accuracy"].reset_index(drop=True)
long_epochs = df[df["Epochs"] == 100]["Accuracy"].reset_index(drop=True)

# Ensure equal lengths
min_len = min(len(short_epochs), len(long_epochs))
if min_len > 0:
    correlation, corr_p = stats.pearsonr(short_epochs.iloc[:min_len], long_epochs.iloc[:min_len])
    print(f"Correlation between 5 and 100 epochs: r = {correlation:.3f}, p-value = {corr_p:.3f}")
    if corr_p < 0.05:
        print("Conclusion: There is a significant correlation between 5-epoch and 100-epoch accuracy.")
    else:
        print("Conclusion: No significant correlation between 5-epoch and 100-epoch accuracy.")
else:
    print("Not enough data to compute correlation between 5 and 100 epochs.")
print()

# Perform pairwise tests between different epoch groups
epoch_values = sorted(df["Epochs"].unique())
for i in range(len(epoch_values) - 1):
    for j in range(i + 1, len(epoch_values)):
        acc1 = df[df["Epochs"] == epoch_values[i]]["Accuracy"]
        acc2 = df[df["Epochs"] == epoch_values[j]]["Accuracy"]
        
        if len(acc1) < 2 or len(acc2) < 2:
            print(f"Skipping {epoch_values[i]} vs {epoch_values[j]} due to insufficient data.")
            continue

        normal1 = stats.shapiro(acc1).pvalue > 0.05
        normal2 = stats.shapiro(acc2).pvalue > 0.05
        
        if normal1 and normal2:
            stat, p_value = stats.ttest_ind(acc1, acc2, equal_var=False)
            test_type = "t-test"
        else:
            stat, p_value = stats.mannwhitneyu(acc1, acc2, alternative="two-sided")
            test_type = "Mann-Whitney U"
        
        print(f"{epoch_values[i]} vs {epoch_values[j]}: {test_type}, p-value = {p_value:.3f}")
        if p_value < 0.05:
            print(f"Conclusion: Significant accuracy difference between {epoch_values[i]} and {epoch_values[j]} epochs.")
        else:
            print(f"Conclusion: No significant difference between {epoch_values[i]} and {epoch_values[j]} epochs.")
        print()

# Visualization
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy vs. Epochs (Boxplot)
sns.boxplot(x=df["Epochs"], y=df["Accuracy"], ax=axes[0], hue=df["Epochs"], palette="Blues", legend=False)
#sns.boxplot(x=df["Epochs"], y=df["Accuracy"], ax=axes[0], palette="Blues")
axes[0].set_title("Accuracy vs. Epochs")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Accuracy")

# Scatter Plot of 5-Epoch vs. 100-Epoch Accuracy
if min_len > 0:
    sns.scatterplot(x=short_epochs.iloc[:min_len], y=long_epochs.iloc[:min_len], ax=axes[1], color="darkblue")
    axes[1].set_title("5-Epoch vs. 100-Epoch Accuracy")
    axes[1].set_xlabel("5-Epoch Accuracy")
    axes[1].set_ylabel("100-Epoch Accuracy")

# Loss vs. Epochs (Boxplot) - Only if "Loss" column exists
if "Loss" in df.columns:
    #sns.boxplot(x=df["Epochs"], y=df["Loss"], ax=axes[2], palette="Reds")
    sns.boxplot(x=df["Epochs"], y=df["Loss"], ax=axes[2], hue=df["Epochs"], palette="Reds", legend=False)

    axes[2].set_title("Loss vs. Epochs")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Loss")
else:
    axes[2].set_visible(False)

plt.tight_layout()
# plt.show()

correlations, recommended = calculate_spearman_correlation(excel_file_path + '.xlsx', reference_epochs=100)

