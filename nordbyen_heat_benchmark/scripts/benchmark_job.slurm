#!/bin/bash
#SBATCH --job-name=benchmark_all_variants
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# SBATCH --partition=rtx3080
# SBATCH --gres=gpu:rtx3080:1

# SBATCH --partition=v100
# SBATCH --gres=gpu:v100:1

# SBATCH --partition=a100
# SBATCH --gres=gpu:a100:1


# Default models if not provided - all four variants
MODELS="${1:-NHITS_Q NHITS_MSE TIMESNET_Q TIMESNET_MSE}"

# Initialize conda properly
eval "$(/apps/python/3.12-conda/bin/conda shell.bash hook)"
conda activate myenv

echo "=========================================="
echo "HEAT CONSUMPTION BENCHMARKING"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Models: $MODELS"
echo "Start Time: $(date)"
echo "Working Dir: $(pwd)"
echo "=========================================="

# Change to project root (parent of nordbyen_heat_benchmark/)
cd "${SLURM_SUBMIT_DIR}/../.."

echo "Changed to project root: $(pwd)"
echo ""

# Run the heat benchmarker (from project root)
python3 nordbyen_heat_benchmark/scripts/run_benchmarker.py --models $MODELS

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
