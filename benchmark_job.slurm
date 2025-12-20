#!/bin/bash
#SBATCH --job-name=benchmark_nhits_timesnet
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# Activate the conda environment
source ~/.bashrc
conda activate myenv

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Run the benchmarker
/home/hpc/iwi5/iwi5389h/.conda/envs/myenv/bin/python3 run_benchmarker.py

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
