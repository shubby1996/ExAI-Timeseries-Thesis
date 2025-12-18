"""
TimesNet Visualization Script.
Loads `results/timesnet_evaluation_metrics.csv` and
`results/timesnet_evaluation_predictions.csv` and creates plots similar to NHiTS visualizations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)


def load_results(results_dir: str):
    metrics_path = os.path.join(results_dir, "timesnet_evaluation_metrics.csv")
    predictions_path = os.path.join(results_dir, "timesnet_evaluation_predictions.csv")
    metrics = pd.read_csv(metrics_path)
    preds = pd.read_csv(predictions_path, parse_dates=["timestamp"]) if os.path.exists(predictions_path) else None
    return metrics, preds


def plot_actual_vs_predicted(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["actual"], label="Actual", color="#2E86AB")
    ax.plot(df["timestamp"], df["predicted"], label="Predicted", color="#A23B72", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Heat Consumption (MW)")
    ax.set_title("TimesNet: Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


def plot_error_distribution(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots()
    sns.histplot(df["abs_error"], bins=50, kde=True, ax=ax, color="#F18F01")
    ax.set_xlabel("Absolute Error (MW)")
    ax.set_title("TimesNet Error Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


def plot_scatter(df: pd.DataFrame, metrics: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots()
    ax.scatter(df["actual"], df["predicted"], alpha=0.5, s=30, color="#4ECDC4", edgecolor="black")
    min_val = min(df["actual"].min(), df["predicted"].min())
    max_val = max(df["actual"].max(), df["predicted"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    # metrics text
    if not metrics.empty:
        mae = metrics["MAE"].values[0]
        rmse = metrics["RMSE"].values[0] if "RMSE" in metrics.columns else None
        r2 = metrics["R²"].values[0] if "R²" in metrics.columns else None
        mape = metrics["MAPE"].values[0] if "MAPE" in metrics.columns else None
        mae_s = f"{mae:.3f}"
        rmse_s = f"{rmse:.3f}" if rmse is not None else "NA"
        r2_s = f"{r2:.3f}" if r2 is not None else "NA"
        mape_s = f"{mape:.2f}%" if mape is not None else "NA"
        text = f"MAE={mae_s}\nRMSE={rmse_s}\nR²={r2_s}\nMAPE={mape_s}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(facecolor="wheat", alpha=0.8))
    ax.set_xlabel("Actual (MW)")
    ax.set_ylabel("Predicted (MW)")
    ax.set_title("TimesNet: Actual vs Predicted Scatter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


def plot_daily_comparison(df: pd.DataFrame, save_path: str, day_offset: int = 10):
    start = day_offset * 24
    end = start + 24
    if end > len(df):
        print("⚠ Not enough data for daily comparison")
        return
    day_df = df.iloc[start:end]
    hours = range(24)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(hours, day_df["actual"], label="Actual", marker="o", color="#2E86AB")
    ax1.plot(hours, day_df["predicted"], label="Predicted", marker="s", color="#A23B72")
    ax1.set_ylabel("Heat (MW)")
    ax1.set_title(f"TimesNet Day {day_offset} Pattern")
    ax1.legend()
    ax2.bar(hours, day_df["error"], color="#F18F01")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Error (MW)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


def plot_error_over_time(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["error"], color="#FF6B6B")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Error (MW)")
    ax.set_title("TimesNet Error Over Time")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RESULTS_DIR = os.path.join(DATA_DIR, "results")

    print("=" * 70)
    print("TimesNet VISUALIZATION")
    print("=" * 70)

    try:
        metrics, preds = load_results(RESULTS_DIR)
        if preds is None:
            raise FileNotFoundError("Predictions file not found in results directory.")

        preds["error"] = preds["actual"] - preds["predicted"]
        preds["abs_error"] = preds["error"].abs()

        plot_actual_vs_predicted(preds, os.path.join(RESULTS_DIR, "timesnet_plot_actual_vs_predicted.png"))
        plot_error_distribution(preds, os.path.join(RESULTS_DIR, "timesnet_plot_error_distribution.png"))
        plot_scatter(preds, metrics, os.path.join(RESULTS_DIR, "timesnet_plot_scatter.png"))
        plot_daily_comparison(preds, os.path.join(RESULTS_DIR, "timesnet_plot_daily_pattern.png"))
        plot_error_over_time(preds, os.path.join(RESULTS_DIR, "timesnet_plot_error_over_time.png"))

        print("\nAll TimesNet visualizations saved to results directory.")
    except FileNotFoundError as e:
        print(f"\nError: Could not find results files. Make sure to run evaluation first.\n{e}")
