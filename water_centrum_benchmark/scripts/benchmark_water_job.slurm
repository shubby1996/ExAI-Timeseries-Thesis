#!/bin/bash
#SBATCH --job-name=benchmark_water
#SBATCH --output=benchmark_water_%j.log
#SBATCH --error=benchmark_water_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# SBATCH --partition=v100
# SBATCH --gres=gpu:v100:1



# SBATCH --partition=rtx3080
# SBATCH --gres=gpu:rtx3080:1

# Default models if not provided - all four variants
MODELS="${1:-NHITS_Q NHITS_MSE TIMESNET_Q TIMESNET_MSE}"

# Initialize conda properly
eval "$(/apps/python/3.12-conda/bin/conda shell.bash hook)"
conda activate myenv

echo "=========================================="
echo "WATER CONSUMPTION BENCHMARKING"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Models: $MODELS"
echo "Start Time: $(date)"
echo "Working Dir: $(pwd)"
echo "=========================================="

# Go to project root
cd /home/hpc/iwi5/iwi5389h/ExAI-Timeseries-Thesis

echo "Changed to project root: $(pwd)"
echo ""

# Run the benchmarker directly using benchmarker.py
python benchmarker.py water_centrum $MODELS

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
