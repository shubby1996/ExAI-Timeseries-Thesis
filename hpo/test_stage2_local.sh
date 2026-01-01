#!/bin/bash
# Quick local test of Stage 2 TimesNet calibration

module load python/3.12-conda
source activate myenv
cd /home/hpc/iwi5/iwi5389h/ExAI-Timeseries-Thesis
export PYTHONPATH="${PYTHONPATH}:/home/hpc/iwi5/iwi5389h/ExAI-Timeseries-Thesis"

echo "============================================================"
echo "STAGE 2 TIMESNET CALIBRATION - LOCAL TEST"
echo "============================================================"

timeout 120 python -c "
import warnings
warnings.filterwarnings('ignore')
import json
import os

# Load Stage 1 best params
stage1_file = 'hpo/results/stage1/water_timesnet_q/best_params.json'
if not os.path.exists(stage1_file):
    print(f'✗ Stage 1 results not found: {stage1_file}')
    print('  Stage 2 requires Stage 1 to complete first')
    exit(1)

with open(stage1_file, 'r') as f:
    stage1_data = json.load(f)

arch_params = stage1_data['best_params']
print(f'✓ Loaded Stage 1 best params:')
print(f'  MAE: {stage1_data[\"best_mae\"]:.6f}')
print(f'  hidden_size: {arch_params[\"hidden_size\"]}')
print(f'  conv_hidden_size: {arch_params[\"conv_hidden_size\"]}')
print(f'  top_k: {arch_params[\"top_k\"]}')
print(f'  lr: {arch_params[\"lr\"]:.6f}')
print(f'  dropout: {arch_params[\"dropout\"]:.4f}')
print()

# Import after params check
from hpo.stage2_calibration import calibrate_timesnet_q

# Test parameters
test_quantiles = [0.1, 0.5, 0.9]  # Default quantiles
csv_path = 'processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv'

print('Testing calibration with quantiles:', test_quantiles)
print('This will take ~60-90s on CPU (2 epochs)...')
print()

picp = calibrate_timesnet_q(csv_path, arch_params, test_quantiles, 'water')

print()
print('✓ SUCCESS! Stage 2 calibration test completed')
print(f'✓ Achieved PICP: {picp:.2f}%')
print(f'  Target: 80%')
print(f'  Calibration error: {abs(picp - 80):.2f}%')
print('============================================================')
"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TEST PASSED - Stage 2 calibration working correctly"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "✗ TEST TIMED OUT - Training took >120s (expected on CPU)"
    echo "  Will run faster on GPU in SLURM job"
else
    echo "✗ TEST FAILED - Exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
