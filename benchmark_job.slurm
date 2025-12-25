#!/bin/bash
#SBATCH --job-name=benchmark_all_variants
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8

# SBATCH --partition=v100
# SBATCH --gres=gpu:v100:1

# SBATCH --partition=a100
# SBATCH --gres=gpu:a100:1


# Default models if not provided - all four variants
MODELS="${1:-NHITS_Q NHITS_MSE TIMESNET_Q TIMESNET_MSE}"

# Activate the conda environment
source ~/.bashrc
conda activate myenv

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Models: $MODELS"
echo "Start Time: $(date)"
echo "=========================================="

# Run the benchmarker
/home/hpc/iwi5/iwi5389h/.conda/envs/myenv/bin/python3 run_benchmarker.py --models $MODELS

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
