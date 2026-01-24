#!/bin/bash
#
# Submit single HPO job to SLURM
#
# Usage:
#   ./hpo/submit_job.sh NHITS_Q heat 50
#   ./hpo/submit_job.sh TFT_Q water_centrum 30
#   ./hpo/submit_job.sh TIMESNET_Q water_tommerby 50
#

MODEL=$1
DATASET=$2
TRIALS=${3:-50}

if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 <MODEL> <DATASET> [TRIALS]"
    echo ""
    echo "Models:   NHITS_Q, TFT_Q, TIMESNET_Q"
    echo "Datasets: heat, water_centrum, water_tommerby"
    echo "Trials:   Default 50"
    echo ""
    echo "Examples:"
    echo "  $0 NHITS_Q heat 50"
    echo "  $0 TFT_Q water_centrum 30"
    exit 1
fi

# Create log directory
mkdir -p hpo/logs

# Submit job and capture output
SBATCH_OUTPUT=$(sbatch \
    --job-name=hpo_${MODEL}_${DATASET} \
    --output=hpo/logs/hpo_${MODEL}_${DATASET}_%j.log \
    --error=hpo/logs/hpo_${MODEL}_${DATASET}_%j.err \
    --time=20:00:00 \
    --partition=a100 \
    --gres=gpu:a100:1 \
    --cpus-per-task=8 \
    --exclude=tg094 \
    --export=ALL,MODEL=$MODEL,DATASET=$DATASET,TRIALS=$TRIALS \
    --wrap="eval \"\$(/apps/python/3.12-conda/bin/conda shell.bash hook)\" && conda activate myenv && python hpo/run_hpo.py --model $MODEL --dataset $DATASET --trials $TRIALS")

# Extract job ID from "Submitted batch job 1234567"
JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to submit job or extract job ID"
    echo "   sbatch output: $SBATCH_OUTPUT"
    exit 1
fi

echo "Submitted batch job $JOB_ID"
echo ""
echo "✅ Submitted HPO job for ${MODEL} on ${DATASET}"
echo "   Job ID: $JOB_ID"
echo "   Trials: $TRIALS"
echo "   Log:    hpo/logs/hpo_${MODEL}_${DATASET}_${JOB_ID}.log"
echo "   Error:  hpo/logs/hpo_${MODEL}_${DATASET}_${JOB_ID}.err"
echo "   Results will be saved to: hpo/results/${MODEL}_${DATASET}/"
echo ""
echo "Monitor with:"
echo "  tail -f hpo/logs/hpo_${MODEL}_${DATASET}_${JOB_ID}.log"
echo "  tail -f hpo/logs/hpo_${MODEL}_${DATASET}_${JOB_ID}.err"
echo "Check status: squeue -j $JOB_ID"
