#!/bin/bash
# Quick test of HPO setup with 1 trial (end-to-end validation)

# Initialize conda
eval "$(/apps/python/3.12-conda/bin/conda shell.bash hook)"
conda activate myenv

echo "Testing NHITS_Q on water_tommerby dataset (1 trial)..."
# python hpo/run_hpo.py --model NHITS_Q --dataset heat --trials 1 --job-id test_quick
# python hpo/run_hpo.py --model TFT_Q --dataset heat --trials 1 --job-id test_quick 
python hpo/run_hpo.py --model NHITS_Q --dataset water_tommerby --trials 1 --job-id test_quick #heat, water_centrum, water_tommerby

echo "" 
echo "✅ Test complete! Check results:"
echo "   hpo/results/NHITS_Q_water_tommerby/best_params_NHITS_Q_water_tommerby_test_quick.json"