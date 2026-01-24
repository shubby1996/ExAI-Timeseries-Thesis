#!/bin/bash
#SBATCH --job-name=hpo_benchmarker
#SBATCH --output=hpo_%j.log
#SBATCH --error=hpo_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# Activate the conda environment
source ~/.bashrc
conda activate myenv

# Model type and number of trials passed as arguments
MODEL_TYPE=$1
N_TRIALS=$2

if [ -z "$MODEL_TYPE" ]; then
    MODEL_TYPE="NHITS"
fi

if [ -z "$N_TRIALS" ]; then
    N_TRIALS=50
fi

echo "Starting HPO for $MODEL_TYPE with $N_TRIALS trials"

# Change to project root (parent of hpo/)
cd "${SLURM_SUBMIT_DIR}/.."

# Run the HPO tuner
/home/hpc/iwi5/iwi5389h/.conda/envs/myenv/bin/python3 hpo/hpo_tuner.py $MODEL_TYPE $N_TRIALS
