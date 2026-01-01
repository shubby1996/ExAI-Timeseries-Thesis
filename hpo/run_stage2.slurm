#!/bin/bash
#SBATCH --job-name=hpo_stage2
#SBATCH --output=hpo/results/stage2/%x_%j.log
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8

# Stage 2: Calibration Optimization
# Usage: sbatch --export=MODEL=NHITS_Q,DATASET=water,TRIALS=20 hpo/run_stage2.slurm

# Load required modules
module load python/3.12-conda

# Activate conda environment
source activate myenv

# Print configuration
echo "============================================================"
echo "STAGE 2: CALIBRATION OPTIMIZATION"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Trials: $TRIALS"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================================"

# Change to project directory
cd /home/hpc/iwi5/iwi5389h/ExAI-Timeseries-Thesis

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/home/hpc/iwi5/iwi5389h/ExAI-Timeseries-Thesis"

# Run Stage 2 calibration
python hpo/stage2_calibration.py \
    --model $MODEL \
    --dataset $DATASET \
    --trials $TRIALS

EXIT_CODE=$?

echo "============================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
