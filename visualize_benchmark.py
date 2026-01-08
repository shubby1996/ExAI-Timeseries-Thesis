import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob
import seaborn as sns
import numpy as np
from properscoring import crps_ensemble


def find_latest_prediction_file(results_dir, model_name):
    """Find the latest prediction file for a given model.
    
    Looks for files matching: {model_name}_predictions_*.csv
    Prefers files with job IDs (numeric) over 'local' files.
    Returns the most recent one based on job ID (numeric).
    """
    pattern = os.path.join(results_dir, f"{model_name}_predictions_*.csv")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        # Fallback to old format without job ID
        old_format = os.path.join(results_dir, f"{model_name}_predictions.csv")
        if os.path.exists(old_format):
            return old_format
        return None
    
    # Separate files with job IDs from 'local' files
    job_id_files = []
    local_files = []
    
    for f in matching_files:
        filename = os.path.basename(f)
        # Extract suffix after last underscore
        suffix = filename.split('_')[-1].replace('.csv', '')
        if suffix.isdigit():
            job_id_files.append((int(suffix), f))
        elif suffix == 'local':
            local_files.append(f)
    
    # Prefer job ID files, return the one with highest job ID (most recent)
    if job_id_files:
        job_id_files.sort(key=lambda x: x[0], reverse=True)
        return job_id_files[0][1]
    
    # Fallback to local file if no job ID files found
    if local_files:
        return local_files[0]
    
    # If nothing found, return most recent alphabetically
    return sorted(matching_files)[-1]


def compute_errors(df):
    df = df.copy()
    df['error'] = df['p50'] - df['actual']
    df['abs_error'] = df['error'].abs()
    df['pct_error'] = 100 * df['error'] / df['actual'].replace(0, 1)
    return df

def compute_metrics(df):
    mae = df['abs_error'].mean()
    rmse = (df['error'] ** 2).mean() ** 0.5
    mape = df['pct_error'].abs().mean()
    picp = ((df['actual'] >= df['p10']) & (df['actual'] <= df['p90'])).mean() * 100
    miw = (df['p90'] - df['p10']).mean()
    
    # Calculate CRPS by approximating samples from quantiles
    crps_values = []
    for _, row in df.iterrows():
        # Skip rows with NaN quantiles
        if pd.isna(row['p10']) or pd.isna(row['p50']) or pd.isna(row['p90']):
            continue
        
        p10, p50, p90 = row['p10'], row['p50'], row['p90']
        
        # Validate quantile ordering
        if p10 >= p50 or p50 >= p90:
            continue
        
        # Approximate std from 80% interval
        std = (p90 - p10) / 2.56
        
        # Skip if std is invalid (should not happen with valid quantiles)
        if std <= 0:
            continue
        
        samples = np.random.normal(p50, std, 100)
        crps_values.append(crps_ensemble(row['actual'], samples))
    
    crps = np.mean(crps_values) if crps_values else np.nan
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'PICP': picp, 'MIW': miw, 'CRPS': crps}

def plot_time_series(df, model_name, save_path):
    plt.figure(figsize=(15, 5))
    df_plot = df.head(168)
    plt.plot(df_plot['timestamp'], df_plot['actual'], label='Actual', color='black', alpha=0.8, linewidth=1.5)
    plt.plot(df_plot['timestamp'], df_plot['p50'], label='Predicted (Median)', color='blue', linewidth=1.2)
    plt.fill_between(df_plot['timestamp'], df_plot['p10'], df_plot['p90'], color='blue', alpha=0.2, label='80% Interval')
    plt.title(f"{model_name} Forecast vs Actual (First 7 Days)", fontsize=14, fontweight='bold')
    plt.xlabel("Timestamp", fontsize=10)
    plt.ylabel("Heat Consumption (Scaled)", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_histogram(df, model_name, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['abs_error'], bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    plt.axvline(df['abs_error'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["abs_error"].mean():.4f}')
    plt.xlabel('Absolute Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name} Absolute Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.hist(df['pct_error'], bins=50, color='#06A77D', alpha=0.7, edgecolor='black')
    plt.axvline(df['pct_error'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["pct_error"].mean():.2f}%')
    plt.xlabel('Percentage Error (%)', fontsize=12)
    plt.title(f'{model_name} Percentage Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_scatter(df, model_name, metrics, save_path):
    plt.figure(figsize=(7, 7))
    plt.scatter(df['actual'], df['p50'], alpha=0.5, s=30, color='#4ECDC4', edgecolors='black', linewidth=0.5)
    min_val = min(df['actual'].min(), df['p50'].min())
    max_val = max(df['actual'].max(), df['p50'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    textstr = '\n'.join([f'{k}: {v:.3f}' for k, v in metrics.items()])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props, family='monospace')
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted (Median)', fontsize=12)
    plt.title(f'{model_name} Actual vs Predicted Scatter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_daily_pattern(df, model_name, save_path, day_offset=10):
    start_idx = day_offset * 24
    end_idx = start_idx + 24
    df_day = df.iloc[start_idx:end_idx].copy()
    if len(df_day) < 24:
        print(f"Not enough data for daily comparison for {model_name}")
        return
    hours = list(range(24))
    plt.figure(figsize=(12, 6))
    plt.plot(hours, df_day['actual'].values, label='Actual', marker='o', color='#2E86AB', linewidth=2.5, markersize=8)
    plt.plot(hours, df_day['p50'].values, label='Predicted', marker='s', color='#A23B72', linewidth=2.5, markersize=8, alpha=0.8)
    plt.fill_between(hours, df_day['p10'].values, df_day['p90'].values, color='#A23B72', alpha=0.2, label='80% Interval')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Heat Consumption (Scaled)', fontsize=12)
    plt.title(f'{model_name} 24-Hour Pattern Comparison\n{df_day["timestamp"].iloc[0].date()}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_over_time(df, model_name, save_path):
    plt.figure(figsize=(15, 4))
    plt.plot(df['timestamp'], df['error'], color='#FF6B6B', linewidth=1.5, alpha=0.7)
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.fill_between(df['timestamp'], 0, df['error'], where=df['error'] >= 0, color='red', alpha=0.3, label='Over-prediction')
    plt.fill_between(df['timestamp'], 0, df['error'], where=df['error'] < 0, color='blue', alpha=0.3, label='Under-prediction')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Prediction Error', fontsize=12)
    plt.title(f'{model_name} Prediction Error Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    # Accept results directory as command-line argument
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    # Ensure results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Visualizing results from: {results_dir}")
    
    sns.set_theme(style="whitegrid")
    models = ["NHITS_Q", "TIMESNET_Q", "TFT_Q", "NHITS_MSE", "TIMESNET_MSE", "TFT_MSE"]
    results = {}
    
    for model in models:
        # Find latest prediction file for this model
        pred_file = find_latest_prediction_file(results_dir, model)
        
        if pred_file is None:
            print(f"File not found for {model} in {results_dir}")
            continue
            
        print(f"Loading {model} from: {pred_file}")
        df = pd.read_csv(pred_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = compute_errors(df)
        metrics = compute_metrics(df)
        results[model] = {'df': df, 'metrics': metrics}
        
        # Plots for each model - save to same results directory
        plot_time_series(df, model, os.path.join(results_dir, f"{model.lower()}_timeseries.png"))
        plot_error_histogram(df, model, os.path.join(results_dir, f"{model.lower()}_error_hist.png"))
        plot_scatter(df, model, metrics, os.path.join(results_dir, f"{model.lower()}_scatter.png"))
        plot_daily_pattern(df, model, os.path.join(results_dir, f"{model.lower()}_daily_pattern.png"))
        plot_error_over_time(df, model, os.path.join(results_dir, f"{model.lower()}_error_over_time.png"))


    # Comparative summary table and metrics bar plots
    if results:
        summary = pd.DataFrame({m: r['metrics'] for m, r in results.items()}).T
        print("\nModel Comparison Metrics:")
        print(summary.round(3))
        summary.to_csv(os.path.join(results_dir, "benchmark_metrics_comparison.csv"))

        # Bar plots for metrics comparison
        metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'PICP', 'MIW', 'CRPS']
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()
        for i, metric in enumerate(metrics_to_plot):
            if metric not in summary.columns:
                continue
            sns.barplot(x=summary.index, y=summary[metric], ax=axes[i], palette="Set2")
            axes[i].set_title(metric)
            axes[i].set_xlabel("")
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
        # Hide unused subplot if any
        if len(metrics_to_plot) < len(axes):
            for j in range(len(metrics_to_plot), len(axes)):
                axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "benchmark_metrics_barplots.png"), dpi=300)
        print(f"Bar plots for metrics saved to {os.path.join(results_dir, 'benchmark_metrics_barplots.png')}")
        plt.close()

        # Box plots for prediction distributions
        pred_data = []
        for model, r in results.items():
            df = r['df']
            for col in ['actual', 'p50', 'p10', 'p90']:
                pred_data.append(pd.DataFrame({
                    'value': df[col],
                    'type': col,
                    'model': model
                }))
        pred_df = pd.concat(pred_data, ignore_index=True)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='type', y='value', hue='model', data=pred_df, palette="Set3")
        plt.title('Box Plot of Actuals and Predictions by Model')
        plt.xlabel('Type')
        plt.ylabel('Heat Consumption (Scaled)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "benchmark_boxplots.png"), dpi=300)
        print(f"Box plots for predictions saved to {os.path.join(results_dir, 'benchmark_boxplots.png')}")
        plt.close()

    # Side-by-side time series plot
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for i, (model, r) in enumerate(results.items()):
        df = r['df'].head(168)
        axes[i].plot(df['timestamp'], df['actual'], label='Actual', color='black', alpha=0.8, linewidth=1.5)
        axes[i].plot(df['timestamp'], df['p50'], label='Predicted (Median)', color='blue', linewidth=1.2)
        axes[i].fill_between(df['timestamp'], df['p10'], df['p90'], color='blue', alpha=0.2, label='80% Interval')
        axes[i].set_title(f"{model}")
        axes[i].set_xlabel("Timestamp")
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].tick_params(axis='x', rotation=45)
    axes[0].set_ylabel("Heat Consumption (Scaled)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "benchmark_comparison_sidebyside.png"), dpi=300)
    print(f"Side-by-side comparison plot saved to {os.path.join(results_dir, 'benchmark_comparison_sidebyside.png')}")
    plt.show()

if __name__ == "__main__":
    main()
