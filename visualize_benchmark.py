import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob
import seaborn as sns
import numpy as np
from properscoring import crps_ensemble
from typing import Dict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


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

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)

def mape_eps(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.sum(np.abs(y_true)), eps)
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)

def mase_denominator(series: np.ndarray, m: int = 24) -> float:
    if len(series) <= m:
        return np.nan
    diffs = np.abs(series[m:] - series[:-m])
    if len(diffs) == 0:
        return np.nan
    return float(np.mean(diffs))

def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    if scale is None or np.isnan(scale) or scale == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

def winkler_score(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray, alpha: float = 0.2) -> float:
    """Winkler score (interval score) - penalizes width and misses. Lower is better."""
    if len(y_true) == 0: 
        return np.nan
    width = y_high - y_low
    miss_below = np.maximum(0, y_low - y_true)
    miss_above = np.maximum(0, y_true - y_high)
    return float(np.mean(width + (2.0 / alpha) * (miss_below + miss_above)))

def calibration_curve(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> Dict[str, np.ndarray]:
    """Calibration curve: empirical vs nominal coverage at different quantile levels."""
    if len(y_true) == 0:
        return {'nominal': np.array([]), 'empirical': np.array([])}
    
    quantile_levels = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    empirical_coverage = []
    
    for q in quantile_levels:
        nominal_cov = 1 - 2 * q
        width = y_high - y_low
        margin = width * q / 0.1  # Scale margin based on quantile level
        emp_low = y_low - margin
        emp_high = y_high + margin
        emp_cov = np.mean((y_true >= emp_low) & (y_true <= emp_high))
        empirical_coverage.append(emp_cov)
    
    return {'nominal': 1 - 2 * quantile_levels, 'empirical': np.array(empirical_coverage)}

def pinball_loss(y_true: np.ndarray, y_pred_quantiles: Dict[float, np.ndarray]) -> float:
    """Calculate average pinball loss across multiple quantiles.
    
    Args:
        y_true: actual values
        y_pred_quantiles: dict mapping quantile levels (e.g., 0.1, 0.5, 0.9) to predicted values
        
    Returns:
        Average pinball loss across all quantiles
    """
    if len(y_true) == 0:
        return np.nan
    
    total_loss = 0.0
    for tau, y_pred in y_pred_quantiles.items():
        errors = y_true - y_pred
        loss = np.where(errors >= 0, tau * errors, (tau - 1) * errors)
        total_loss += np.mean(loss)
    
    return total_loss / len(y_pred_quantiles)

def compute_metrics(df, is_mse=False):
    mae = df['abs_error'].mean()
    rmse = (df['error'] ** 2).mean() ** 0.5
    mape = df['pct_error'].abs().mean()
    # mape_eps_val = mape_eps(df['actual'].values, df['p50'].values)
    smape_val = smape(df['actual'].values, df['p50'].values)
    wape_val = wape(df['actual'].values, df['p50'].values)
    mase_scale = mase_denominator(df['actual'].values, m=24)
    mase_val = mase(df['actual'].values, df['p50'].values, mase_scale)
    
    # For MSE models, skip probabilistic metrics
    if is_mse:
        picp = np.nan
        miw = np.nan
        winkler = np.nan
        crps = np.nan
        pinball = np.nan
    else:
        picp = ((df['actual'] >= df['p10']) & (df['actual'] <= df['p90'])).mean() * 100
        miw = (df['p90'] - df['p10']).mean()
        
        # Calculate pinball loss from quantiles
        pinball = pinball_loss(
            df['actual'].values,
            {0.1: df['p10'].values, 0.5: df['p50'].values, 0.9: df['p90'].values}
        )
        
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
        winkler = winkler_score(df['actual'].values, df['p10'].values, df['p90'].values) if (df['p10'].notna().any() and df['p90'].notna().any()) else np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        # 'MAPE_EPS': mape_eps_val,
        'sMAPE': smape_val,
        'WAPE': wape_val,
        'MASE': mase_val,
        'Pinball': pinball,
        'PICP': picp,
        'MIW': miw,
        'Winkler': winkler,
        'CRPS': crps
    }

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

def plot_calibration(results_dict, results_dir):
    """Plot calibration curves: empirical vs nominal coverage for all models."""
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6 * len(results_dict), 5))
    if len(results_dict) == 1:
        axes = [axes]
    
    for i, (model, r) in enumerate(results_dict.items()):
        df = r['df']
        cal = calibration_curve(df['actual'].values, df['p10'].values, df['p90'].values)
        
        ax = axes[i]
        ax.plot(cal['nominal'], cal['empirical'], 'o-', linewidth=2, markersize=8, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')
        ax.set_xlabel('Nominal Coverage', fontsize=11)
        ax.set_ylabel('Empirical Coverage', fontsize=11)
        ax.set_title(f'{model} Calibration')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'calibration_curves.png'), dpi=300)
    print(f"Calibration plot saved to {os.path.join(results_dir, 'calibration_curves.png')}")
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
    models = ["NHITS_Q", "TIMESNET_Q", "TFT_Q", "NAMLSS", "NHITS_MSE", "TIMESNET_MSE", "TFT_MSE"]
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
        is_mse = 'MSE' in model
        metrics = compute_metrics(df, is_mse=is_mse)
        results[model] = {'df': df, 'metrics': metrics}
        
        # # Plots for each model - save to same results directory
        # plot_time_series(df, model, os.path.join(results_dir, f"{model.lower()}_timeseries.png"))
        # plot_error_histogram(df, model, os.path.join(results_dir, f"{model.lower()}_error_hist.png"))
        # plot_scatter(df, model, metrics, os.path.join(results_dir, f"{model.lower()}_scatter.png"))
        # plot_daily_pattern(df, model, os.path.join(results_dir, f"{model.lower()}_daily_pattern.png"))
        # plot_error_over_time(df, model, os.path.join(results_dir, f"{model.lower()}_error_over_time.png"))


    # Comparative summary table and metrics bar plots
    if results:
        summary = pd.DataFrame({m: r['metrics'] for m, r in results.items()}).T
        print("\nModel Comparison Metrics:")
        print(summary.round(3))
        summary.to_csv(os.path.join(results_dir, "benchmark_metrics_comparison.csv"))

        # Bar plots for metrics comparison - separated into point forecast and probabilistic
        point_forecast_metrics = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'WAPE', 'MASE']
        probabilistic_metrics = ['Pinball', 'Winkler', 'PICP', 'MIW', 'CRPS']
        
        # Plot point forecast metrics (2 rows x 3 cols)
        cols = 3
        rows = math.ceil(len(point_forecast_metrics) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = axes.flatten()
        for i, metric in enumerate(point_forecast_metrics):
            if metric not in summary.columns:
                axes[i].set_visible(False)
                continue
            if summary[metric].isna().all():
                axes[i].set_visible(False)
                continue
            metric_data = summary[metric].dropna()
            if len(metric_data) == 0:
                axes[i].set_visible(False)
                continue
            sns.barplot(x=metric_data.index, y=metric_data.values, ax=axes[i], palette="Set2")
            axes[i].set_title(metric, fontsize=12, fontweight='bold')
            axes[i].set_xlabel("")
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        for j in range(len(point_forecast_metrics), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "benchmark_metrics_point_forecast.png"), dpi=300)
        print(f"Point forecast metrics plot saved to {os.path.join(results_dir, 'benchmark_metrics_point_forecast.png')}")
        plt.close()
        
        # Plot probabilistic metrics (1 row x 4 cols with thinner bars)
        prob_metrics_to_plot = [m for m in probabilistic_metrics if m in summary.columns and not summary[m].isna().all()]
        if prob_metrics_to_plot:
            n_prob = len(prob_metrics_to_plot)
            fig, axes = plt.subplots(1, n_prob, figsize=(4.5 * n_prob, 5))
            if n_prob == 1:
                axes = [axes]
            
            for i, metric in enumerate(prob_metrics_to_plot):
                metric_data = summary[metric].dropna()
                if len(metric_data) == 0:
                    axes[i].set_visible(False)
                    continue
                sns.barplot(x=metric_data.index, y=metric_data.values, ax=axes[i], palette="Set3", width=0.6)
                axes[i].set_title(metric, fontsize=12, fontweight='bold')
                axes[i].set_xlabel("")
                axes[i].set_ylabel(metric)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "benchmark_metrics_probabilistic.png"), dpi=300)
            print(f"Probabilistic metrics plot saved to {os.path.join(results_dir, 'benchmark_metrics_probabilistic.png')}")
            plt.close()

        # Calibration curves
        plot_calibration(results, results_dir)

        # ===== INTERACTIVE PLOTLY VISUALIZATIONS =====
        print("\n" + "="*70)
        print("Generating Interactive Plotly Visualizations...")
        print("="*70)
        
        # Create directory for interactive plots
        interactive_dir = os.path.join(results_dir, "interactive_plots")
        os.makedirs(interactive_dir, exist_ok=True)
        
        # Generate interactive plots for each model
        for model, r in results.items():
            df = r['df']
            metrics = r['metrics']
            
            # Time series plot
            ts_path = os.path.join(interactive_dir, f"{model.lower()}_timeseries.html")
            plot_interactive_timeseries(df, model, ts_path)
            print(f"  ✓ {model}: Interactive time series saved")
            
            # Error distribution plot
            err_path = os.path.join(interactive_dir, f"{model.lower()}_error_distribution.html")
            plot_interactive_error_distribution(df, model, err_path)
            print(f"  ✓ {model}: Interactive error distribution saved")
            
            # Scatter plot
            scatter_path = os.path.join(interactive_dir, f"{model.lower()}_scatter.html")
            plot_interactive_scatter(df, model, metrics, scatter_path)
            print(f"  ✓ {model}: Interactive scatter plot saved")
        
        # Metrics comparison
        plot_interactive_metrics_comparison(results, interactive_dir)
        
        print(f"\nAll interactive plots saved to: {interactive_dir}")

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

def plot_interactive_timeseries(df, model_name, save_path):
    """Create interactive time series plot with Plotly."""
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['actual'],
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2),
        hovertemplate='<b>Actual</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    # Add predicted median
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['p50'],
        mode='lines',
        name='Predicted (Median)',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Predicted</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    # Add confidence interval as filled area
    if df['p10'].notna().any() and df['p90'].notna().any():
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['p90'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['p10'],
            fillcolor='rgba(31, 119, 180, 0.2)',
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='80% Interval',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f'{model_name} - Full Test Set Forecast',
        xaxis_title='Timestamp',
        yaxis_title='Value (Scaled)',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )
    
    fig.write_html(save_path)
    # Also save as JSON for notebook embedding
    json_path = save_path.replace('.html', '.json')
    fig.write_json(json_path)
    return fig

def plot_interactive_error_distribution(df, model_name, save_path):
    """Create interactive error distribution plots with Plotly."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Absolute Error Distribution', 'Percentage Error Distribution')
    )
    
    # Absolute error histogram
    fig.add_trace(
        go.Histogram(
            x=df['abs_error'],
            name='Absolute Error',
            nbinsx=50,
            marker_color='#FF9999',
            hovertemplate='<b>Absolute Error</b><br>Range: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Percentage error histogram
    fig.add_trace(
        go.Histogram(
            x=df['pct_error'],
            name='Percentage Error',
            nbinsx=50,
            marker_color='#66B2FF',
            hovertemplate='<b>Percentage Error</b><br>Range: %{x}%<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Absolute Error', row=1, col=1)
    fig.update_xaxes(title_text='Percentage Error (%)', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)
    
    fig.update_layout(
        title_text=f'{model_name} - Error Distributions',
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.write_html(save_path)
    json_path = save_path.replace('.html', '.json')
    fig.write_json(json_path)
    return fig

def plot_interactive_scatter(df, model_name, metrics, save_path):
    """Create interactive scatter plot of actual vs predicted."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['actual'],
        y=df['p50'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['abs_error'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Absolute Error'),
            line=dict(width=0.5, color='white')
        ),
        text=[f'Actual: {a:.4f}<br>Predicted: {p:.4f}<br>Error: {e:.4f}' 
              for a, p, e in zip(df['actual'], df['p50'], df['abs_error'])],
        hovertemplate='%{text}<extra></extra>',
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(df['actual'].min(), df['p50'].min())
    max_val = max(df['actual'].max(), df['p50'].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Perfect Prediction',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'{model_name} - Actual vs Predicted',
        xaxis_title='Actual Value',
        yaxis_title='Predicted Value (Median)',
        height=500,
        width=600,
        template='plotly_white',
        hovermode='closest'
    )
    
    fig.write_html(save_path)
    json_path = save_path.replace('.html', '.json')
    fig.write_json(json_path)
    return fig

def plot_interactive_metrics_comparison(results_dict, results_dir):
    """Create interactive metrics comparison using Plotly."""
    # Prepare data
    summary = pd.DataFrame({m: r['metrics'] for m, r in results_dict.items()}).T
    summary = summary.reset_index().rename(columns={'index': 'Model'})
    
    # Select point forecast metrics (applicable to all models)
    point_metrics = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'WAPE', 'MASE']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=point_metrics,
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    colors = px.colors.qualitative.Set2
    
    for idx, metric in enumerate(point_metrics):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
        if metric in summary.columns:
            metric_data = summary[['Model', metric]].dropna()
            
            fig.add_trace(
                go.Bar(
                    x=metric_data['Model'],
                    y=metric_data[metric],
                    name=metric,
                    marker_color=colors[idx % len(colors)],
                    hovertemplate='<b>%{x}</b><br>' + metric + ': %{y:.4f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            fig.update_yaxes(title_text=metric, row=row, col=col)
    
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(
        title_text='Model Metrics Comparison (Point Forecast)',
        height=700,
        showlegend=False,
        template='plotly_white'
    )
    
    html_path = os.path.join(results_dir, 'interactive_metrics_comparison.html')
    fig.write_html(html_path)
    json_path = html_path.replace('.html', '.json')
    fig.write_json(json_path)
    print(f"Interactive metrics comparison saved to {html_path}")
    return fig



if __name__ == "__main__":
    main()
