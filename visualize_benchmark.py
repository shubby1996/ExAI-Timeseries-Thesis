import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_model_results(model_name, ax):
    path = f"results/{model_name}_predictions.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Use a subset for clearer visualization (e.g., first 7 days of test set)
    df_plot = df.head(168) 
    
    ax.plot(df_plot['timestamp'], df_plot['actual'], label='Actual', color='black', alpha=0.8, linewidth=1.5)
    ax.plot(df_plot['timestamp'], df_plot['p50'], label='Predicted (Median)', color='blue', linewidth=1.2)
    ax.fill_between(df_plot['timestamp'], df_plot['p10'], df_plot['p90'], color='blue', alpha=0.2, label='80% Interval')
    
    ax.set_title(f"{model_name} Forecast vs Actual", fontsize=14, fontweight='bold')
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.set_ylabel("Heat Consumption (Scaled)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)

def main():
    sns.set_theme(style="whitegrid")
    models = ["NHITS", "TIMESNET"]
    
    fig, axes = plt.subplots(len(models), 1, figsize=(15, 6 * len(models)), constrained_layout=True)
    
    if len(models) == 1:
        axes = [axes]
        
    for i, model in enumerate(models):
        plot_model_results(model, axes[i])
        
    output_path = "results/benchmark_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
