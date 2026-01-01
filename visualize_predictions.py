"""
TFT Model Visualization Script for Nordbyen Heat Consumption Forecasting.

This script creates various visualizations for model performance analysis:
1. Actual vs Predicted time series
2. Error distribution histogram
3. Scatter plot (actual vs predicted)
4. Daily pattern comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_evaluation_results(results_dir: str):
    """Load evaluation results from CSV files."""
    metrics_path = os.path.join(results_dir, "evaluation_metrics.csv")
    predictions_path = os.path.join(results_dir, "evaluation_predictions.csv")
    
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path, parse_dates=['timestamp'])
    
    return metrics, predictions


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, save_path: str, n_samples: int = 500):
    """Plot actual vs predicted time series."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Use subset for clarity
    df_subset = predictions_df.head(n_samples)
    
    ax.plot(df_subset['timestamp'], df_subset['actual'], 
            label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
    
    # Use p50 as predicted if available, else use 'predicted'
    pred_col = 'p50' if 'p50' in df_subset.columns else 'predicted'
    ax.plot(df_subset['timestamp'], df_subset[pred_col], 
            label='Predicted (Median)', color='#A23B72', linewidth=2, alpha=0.8, linestyle='--')
    
    # Add confidence interval if available (check for p10/p90 or predicted_low/predicted_high)
    if 'p10' in df_subset.columns and 'p90' in df_subset.columns:
        ax.fill_between(df_subset['timestamp'], 
                        df_subset['p10'], 
                        df_subset['p90'],
                        color='#A23B72', alpha=0.2, label='10-90% Confidence Interval')
    elif 'predicted_low' in df_subset.columns and 'predicted_high' in df_subset.columns:
        ax.fill_between(df_subset['timestamp'], 
                        df_subset['predicted_low'], 
                        df_subset['predicted_high'],
                        color='#A23B72', alpha=0.2, label='10-90% Confidence Interval')
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Heat Consumption (MW)', fontsize=12, fontweight='bold')
    ax.set_title('TFT Model: Actual vs Predicted Heat Consumption\n(First {} Hours of Test Set)'.format(n_samples),
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_error_distribution(predictions_df: pd.DataFrame, save_path: str):
    """Plot error distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute error histogram
    axes[0].hist(predictions_df['abs_error'], bins=50, color='#F18F01', 
                 alpha=0.7, edgecolor='black')
    axes[0].axvline(predictions_df['abs_error'].mean(), color='red', 
                    linestyle='--', linewidth=2, label=f'Mean: {predictions_df["abs_error"].mean():.4f}')
    axes[0].set_xlabel('Absolute Error (MW)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Absolute Errors', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Percentage error histogram
    axes[1].hist(predictions_df['pct_error'], bins=50, color='#06A77D', 
                 alpha=0.7, edgecolor='black')
    axes[1].axvline(predictions_df['pct_error'].mean(), color='red', 
                    linestyle='--', linewidth=2, label=f'Mean: {predictions_df["pct_error"].mean():.2f}%')
    axes[1].set_xlabel('Percentage Error (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Percentage Errors', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_scatter(predictions_df: pd.DataFrame, metrics: pd.DataFrame, save_path: str):
    """Plot scatter of actual vs predicted with perfect prediction line."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(predictions_df['actual'], predictions_df['predicted'], 
               alpha=0.5, s=30, color='#4ECDC4', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
    max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    
    # Add metrics text box
    r2 = metrics['R²'].values[0]
    mae = metrics['MAE'].values[0]
    rmse = metrics['RMSE'].values[0]
    mape = metrics['MAPE'].values[0]
    
    textstr = f'R² = {r2:.4f}\nMAE = {mae:.4f} MW\nRMSE = {rmse:.4f} MW\nMAPE = {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.set_xlabel('Actual Heat Consumption (MW)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Heat Consumption (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Actual vs Predicted Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_daily_comparison(predictions_df: pd.DataFrame, save_path: str, day_offset: int = 10):
    """Plot actual vs predicted for a sample day."""
    # Extract one day worth of data (24 hours)
    start_idx = day_offset * 24
    end_idx = start_idx + 24
    df_day = predictions_df.iloc[start_idx:end_idx].copy()
    
    if len(df_day) < 24:
        print("⚠ Not enough data for daily comparison")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series comparison with intervals
    hours = list(range(24))
    
    # Use p50 as predicted if available, else use 'predicted'
    pred_col = 'p50' if 'p50' in df_day.columns else 'predicted'
    
    ax1.plot(hours, df_day['actual'].values, 
             label='Actual', marker='o', color='#2E86AB', linewidth=2.5, markersize=8)
    ax1.plot(hours, df_day[pred_col].values, 
             label='Predicted', marker='s', color='#A23B72', linewidth=2.5, markersize=8, alpha=0.8)
    
    # Add confidence interval (check for p10/p90 or predicted_low/predicted_high)
    if 'p10' in df_day.columns and 'p90' in df_day.columns:
        ax1.fill_between(hours, 
                        df_day['p10'].values, 
                        df_day['p90'].values,
                        color='#A23B72', alpha=0.2, label='10-90% Confidence')
    elif 'predicted_low' in df_day.columns and 'predicted_high' in df_day.columns:
        ax1.fill_between(hours, 
                        df_day['predicted_low'].values, 
                        df_day['predicted_high'].values,
                        color='#A23B72', alpha=0.2, label='10-90% Confidence Interval')
    else:
        # Fallback to error shading if no CI
        ax1.fill_between(hours, df_day['actual'].values, df_day['predicted'].values, 
                          alpha=0.3, color='gray', label='Prediction Error')
    
    ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Heat Consumption (MW)', fontsize=12, fontweight='bold')
    ax1.set_title(f'24-Hour Pattern Comparison\n{df_day["timestamp"].iloc[0].date()}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(hours)
    
    # Plot 2: Error over the day
    ax2.bar(hours, df_day['error'].values, color='#F18F01', alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prediction Error (MW)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Error by Hour', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(hours)
    
    # Add statistics
    mae_day = df_day['abs_error'].mean()
    rmse_day = np.sqrt((df_day['error']**2).mean())
    textstr = f'Daily MAE = {mae_day:.4f} MW\nDaily RMSE = {rmse_day:.4f} MW'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_error_over_time(predictions_df: pd.DataFrame, save_path: str):
    """Plot how error evolves over time."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.plot(predictions_df['timestamp'], predictions_df['error'], 
            color='#FF6B6B', linewidth=1.5, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.fill_between(predictions_df['timestamp'], 0, predictions_df['error'], 
                     where=predictions_df['error'] >= 0, color='red', alpha=0.3, label='Over-prediction')
    ax.fill_between(predictions_df['timestamp'], 0, predictions_df['error'], 
                     where=predictions_df['error'] < 0, color='blue', alpha=0.3, label='Under-prediction')
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = r"c:\Uni Stuff\Semester 5\Thesis_SI\ShubhamThesis\data"
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    
    print("=" * 70)
    print("TFT MODEL VISUALIZATION")
    print("=" * 70)
    
    # Load results
    print("\n[1/2] Loading evaluation results...")
    metrics, predictions = load_evaluation_results(RESULTS_DIR)
    print(f"  ✓ Loaded {len(predictions)} predictions")
    print(f"  ✓ Metrics: MAE={metrics['MAE'].values[0]:.4f}, R²={metrics['R²'].values[0]:.4f}")
    
    # Generate visualizations
    print("\n[2/2] Generating visualizations...")
    
    plot_actual_vs_predicted(
        predictions, 
        os.path.join(RESULTS_DIR, "plot_actual_vs_predicted.png"),
        n_samples=500
    )
    
    plot_error_distribution(
        predictions,
        os.path.join(RESULTS_DIR, "plot_error_distribution.png")
    )
    
    plot_scatter(
        predictions,
        metrics,
        os.path.join(RESULTS_DIR, "plot_scatter.png")
    )
    
    plot_daily_comparison(
        predictions,
        os.path.join(RESULTS_DIR, "plot_daily_pattern.png"),
        day_offset=10
    )
    
    plot_error_over_time(
        predictions,
        os.path.join(RESULTS_DIR, "plot_error_over_time.png")
    )
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ All plots saved to: {RESULTS_DIR}")
