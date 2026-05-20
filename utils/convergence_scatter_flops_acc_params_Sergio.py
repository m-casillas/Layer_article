import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load generation statistics from CSV.
    """
    return pd.read_csv(csv_path)


def plot_with_std(x, mean, std, title, xlabel, ylabel, color="C0", marker="o"):
    """
    Generic plotting function for mean ± std curves.
    """
    plt.figure(figsize=(9, 6))

    plt.plot(
        x,
        mean,
        marker=marker,
        markevery=max(len(x)//20, 1),
        linewidth=0.5,
        color=color,
        label=ylabel
    )

    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2,
        color=color
    )

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_accuracy(df: pd.DataFrame):
    plot_with_std(
        df["Generation"],
        df["mean_accuracy"],
        df["std_accuracy"],
        title="Accuracy per Generation",
        xlabel="Generation",
        ylabel="Mean Accuracy",
        color="C0",
        marker="o"
    )


def plot_flops(df: pd.DataFrame):
    plot_with_std(
        df["Generation"],
        df["mean_flops"],
        df["std_flops"],
        title="FLOPs per Generation",
        xlabel="Generation",
        ylabel="Mean FLOPs",
        color="C1",
        marker="s"
    )


def plot_params(df: pd.DataFrame):
    plot_with_std(
        df["Generation"],
        df["mean_params"],
        df["std_params"],
        title="Parameters per Generation",
        xlabel="Generation",
        ylabel="Mean Parameters",
        color="C2",
        marker="^"
    )


def main():
    csv_file = "generation_status.csv"

    df = load_data(csv_file)

    plot_accuracy(df)
    plot_flops(df)
    plot_params(df)


if __name__ == "__main__":
    main()