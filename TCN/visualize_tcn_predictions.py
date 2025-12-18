"""
Visualize TCN Fresh Model Predictions
Creates plots comparing actual vs predicted values
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Load results
print("Loading TCN results...")
predictions_path = os.path.join(RESULTS_DIR, "tcn_fresh_predictions.csv")
metrics_path = os.path.join(RESULTS_DIR, "tcn_fresh_metrics.csv")

if not os.path.exists(predictions_path):
    print(f"Error: Could not find {predictions_path}")
    print("Make sure to run evaluation first: python evaluate_tcn_fresh.py")
    exit(1)

df = pd.read_csv(predictions_path)
metrics = pd.read_csv(metrics_path)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Loaded {len(df)} predictions")
print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Extract metrics
mae = metrics['MAE'].values[0]
rmse = metrics['RMSE'].values[0]
mape = metrics['MAPE'].values[0]
r2 = metrics['R2'].values[0]

print(f"\nMetrics:")
print(f"  MAE:  {mae:.4f} MW")
print(f"  RMSE: {rmse:.4f} MW")
print(f"  MAPE: {mape:.2f}%")
print(f"  R2:   {r2:.4f}")

# Create figure
print("\nCreating visualization...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Full time series
ax1 = axes[0]
ax1.plot(df['timestamp'], df['actual'], label='Actual', color='black', linewidth=1.5, alpha=0.8)
ax1.plot(df['timestamp'], df['predicted'], label='TCN Predicted', color='#2E86AB', linewidth=1.2, alpha=0.9)
ax1.set_xlabel('Time', fontsize=11)
ax1.set_ylabel('Heat Consumption (MW)', fontsize=11)
ax1.set_title('TCN Model: Actual vs Predicted Heat Consumption (Full Test Set)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add metrics text box
metrics_text = f'MAE: {mae:.3f} MW\nRMSE: {rmse:.3f} MW\nMAPE: {mape:.2f}%\nR²: {r2:.3f}'
ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Zoomed view (first 7 days)
ax2 = axes[1]
zoom_hours = 7 * 24  # 7 days
df_zoom = df.head(zoom_hours)

ax2.plot(df_zoom['timestamp'], df_zoom['actual'], label='Actual', 
         color='black', linewidth=2, marker='o', markersize=3, alpha=0.8)
ax2.plot(df_zoom['timestamp'], df_zoom['predicted'], label='TCN Predicted', 
         color='#2E86AB', linewidth=2, marker='s', markersize=3, alpha=0.9)
ax2.set_xlabel('Time', fontsize=11)
ax2.set_ylabel('Heat Consumption (MW)', fontsize=11)
ax2.set_title('TCN Model: Detailed View (First 7 Days)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Save
output_path = os.path.join(RESULTS_DIR, "tcn_fresh_plot_actual_vs_predicted.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Create error distribution plot
print("\nCreating error distribution plot...")
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Calculate errors
df['error'] = df['actual'] - df['predicted']
df['abs_error'] = df['error'].abs()
df['pct_error'] = (df['abs_error'] / df['actual']) * 100

# Plot 1: Error distribution
ax1 = axes2[0]
ax1.hist(df['error'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax1.axvline(x=df['error'].mean(), color='green', linestyle='--', linewidth=2, 
            label=f'Mean Error: {df["error"].mean():.3f} MW')
ax1.set_xlabel('Prediction Error (MW)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('TCN Error Distribution', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: Absolute error over time
ax2 = axes2[1]
ax2.plot(df['timestamp'], df['abs_error'], color='#A23B72', linewidth=1, alpha=0.7)
ax2.axhline(y=mae, color='red', linestyle='--', linewidth=2, label=f'MAE: {mae:.3f} MW')
ax2.set_xlabel('Time', fontsize=11)
ax2.set_ylabel('Absolute Error (MW)', fontsize=11)
ax2.set_title('TCN Absolute Error Over Time', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
error_output_path = os.path.join(RESULTS_DIR, "tcn_fresh_error_analysis.png")
plt.savefig(error_output_path, dpi=300, bbox_inches='tight')
print(f"Error plot saved to: {error_output_path}")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"\nGenerated plots:")
print(f"  1. {output_path}")
print(f"  2. {error_output_path}")
