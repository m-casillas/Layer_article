import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------
# Load CSV data
# -------------------------------------------------
def load_data(file_path):
    return pd.read_csv(file_path)


# -------------------------------------------------
# Compute moving average tendency
# -------------------------------------------------
def compute_tendency(values, window=10):
    tendency = pd.Series(values).rolling(window, center=True).mean().to_numpy()
    return tendency


# -------------------------------------------------
# Find best point (max or min depending on metric)
# -------------------------------------------------
def find_best_point(values, generations, mode="max"):

    if mode == "max":
        idx = np.nanargmax(values)
    else:
        idx = np.nanargmin(values)

    return generations[idx], values[idx], idx


# -------------------------------------------------
# Generic plotting function
# -------------------------------------------------
def plot_metric(generations, mean_values, std_values, metric_name, mode="max"):

    tendency = compute_tendency(mean_values)

    best_gen, best_val, idx = find_best_point(tendency, generations, mode)

    plt.figure(figsize=(8,6))

    # Error bars
    plt.errorbar(
        generations,
        mean_values,
        yerr=std_values,
        fmt='o',
        color='blue',
        ecolor='lightcoral',
        capsize=4,
        elinewidth=1,
        alpha=0.6,
        label=f"Mean {metric_name} with Std Dev",
        zorder=1
    )

    # Tendency
    #plt.plot(generations,tendency, 'k--', linewidth=3,label="Tendency",zorder=3)

    # Best point
    plt.scatter(
        best_gen,
        best_val,
        color='red',
        marker='x',
        s=200,
        label=f"Best {metric_name}",
        zorder=4
    )

    # Guide lines
    plt.axvline(best_gen, linestyle='--', color='black', zorder=0)
    plt.axhline(best_val, linestyle='--', color='black', zorder=0)

    # Axis labels
    plt.xlabel("Generation")
    plt.ylabel(metric_name)

    # Limits
    y_min = np.nanmin(mean_values - std_values)
    y_max = np.nanmax(mean_values + std_values)
    plt.ylim(y_min, y_max)

    # Grid
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Main program
# -------------------------------------------------
def main():

    data = load_data("generation_status.csv")
    #data = data.head(600)
    generations = data["Generation"].values

    # Accuracy
    plot_metric(
        generations,
        data["mean_accuracy"].values,
        data["std_accuracy"].values,
        "Accuracy",
        mode="max"
    )

    # FLOPs
    plot_metric(
        generations,
        data["mean_flops"].values,
        data["std_flops"].values,
        "FLOPs",
        mode="min"
    )

    # Params
    plot_metric(
        generations,
        data["mean_params"].values,
        data["std_params"].values,
        "#Params",
        mode="min"
    )


# Run program
if __name__ == "__main__":
    main()