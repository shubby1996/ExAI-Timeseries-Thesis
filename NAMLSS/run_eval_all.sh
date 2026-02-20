#!/bin/bash
#
# Run eval_tsnamlss.py (post-benchmarking explainability) for all three datasets.
#
# Usage (from the NAMLSS/ directory):
#   ./run_eval_all.sh
#   ./run_eval_all.sh --device cuda   # override device for all runs
#
# Outputs (relative to project root):
#   eval_out/nordbyen_NAMLSS/
#   eval_out/centrum_NAMLSS/
#   eval_out/tommerby_NAMLSS/
#
# Each output directory contains:
#   effect_importance_stream_level.json
#   mu_importance_by_horizon_norm_y.npy
#   mu_importance_by_horizon_per_cov_norm_y.npy
#   rawsig_importance_by_horizon_norm_y.npy
#   rawsig_importance_by_horizon_per_cov_norm_y.npy
#   rawsig_importance_by_horizon_norm_rawsig.npy
#   rawsig_importance_by_horizon_per_cov_norm_rawsig.npy
#

set -e  # Exit immediately on error

# Allow device override via first argument (default: cpu)
DEVICE="${1:---device cpu}"
# If user passed --device cuda, pass it through; otherwise default to cpu
if [[ "$1" == "--device" && -n "$2" ]]; then
    DEVICE="--device $2"
    shift 2
else
    DEVICE="--device cpu"
fi

# Resolve project root (one level above this script's directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=========================================="
echo "  NAMLSS Explainability Evaluation"
echo "  Project root: $PROJECT_ROOT"
echo "  Device: $DEVICE"
echo "=========================================="
echo ""

# -----------------------------------------------
# 1. Heat (Nordbyen)
#    Test period: 2020-01-01 → 2020-11-14
# -----------------------------------------------
echo "[1/3] Heat (Nordbyen) ..."
python3 "$SCRIPT_DIR/eval_tsnamlss.py" \
    --csv_path            "$PROJECT_ROOT/processing/nordbyen_processing/nordbyen_features_engineered.csv" \
    --ckpt                "$PROJECT_ROOT/models/nordbyen_heat/NAMLSS.pt" \
    --preprocessing_state "$PROJECT_ROOT/models/nordbyen_heat/NAMLSS_preprocessing_state.pkl" \
    --test_start_str 2020-01-01 \
    --test_end_str   2020-11-14 \
    $DEVICE \
    --out_dir "$PROJECT_ROOT/eval_out/nordbyen_NAMLSS"

echo ""
echo "  ✓ Saved to: eval_out/nordbyen_NAMLSS/"
echo ""

# -----------------------------------------------
# 2. Water (Centrum)
#    Test period: 2020-05-07 → 2020-11-14
# -----------------------------------------------
echo "[2/3] Water (Centrum) ..."
python3 "$SCRIPT_DIR/eval_tsnamlss.py" \
    --csv_path            "$PROJECT_ROOT/processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv" \
    --ckpt                "$PROJECT_ROOT/models/water_centrum/NAMLSS.pt" \
    --preprocessing_state "$PROJECT_ROOT/models/water_centrum/NAMLSS_preprocessing_state.pkl" \
    --test_start_str 2020-05-07 \
    --test_end_str   2020-11-14 \
    $DEVICE \
    --out_dir "$PROJECT_ROOT/eval_out/centrum_NAMLSS"

echo ""
echo "  ✓ Saved to: eval_out/centrum_NAMLSS/"
echo ""

# -----------------------------------------------
# 3. Water (Tommerby)
#    Test period: 2020-05-07 → 2020-11-14
# -----------------------------------------------
echo "[3/3] Water (Tommerby) ..."
python3 "$SCRIPT_DIR/eval_tsnamlss.py" \
    --csv_path            "$PROJECT_ROOT/processing/tommerby_processing/tommerby_features_engineered_from_2018-04-01.csv" \
    --ckpt                "$PROJECT_ROOT/models/water_tommerby/NAMLSS.pt" \
    --preprocessing_state "$PROJECT_ROOT/models/water_tommerby/NAMLSS_preprocessing_state.pkl" \
    --test_start_str 2020-05-07 \
    --test_end_str   2020-11-14 \
    $DEVICE \
    --out_dir "$PROJECT_ROOT/eval_out/tommerby_NAMLSS"

echo ""
echo "  ✓ Saved to: eval_out/tommerby_NAMLSS/"
echo ""

echo "=========================================="
echo "  All done! Explainability artifacts in:"
echo "    $PROJECT_ROOT/eval_out/"
echo "=========================================="
