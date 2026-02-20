#!/bin/bash
#
# Run interpret_tsnamlss.py for all three datasets (post-eval explainability).
#
# Usage (from the NAMLSS/ directory):
#   ./run_interpret_all.sh
#   ./run_interpret_all.sh --device cuda   # override device for all runs
#
# Outputs (relative to project root):
#   interp_out/nordbyen_NAMLSS/
#   interp_out/centrum_NAMLSS/
#   interp_out/tommerby_NAMLSS/
#
# Each output directory contains interactive HTML plots and effect artifacts.
#
set -e

# Allow device override via first argument (default: cpu)
DEVICE="${1:---device cpu}"
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
echo "  NAMLSS Interpretation (all datasets)"
echo "  Project root: $PROJECT_ROOT"
echo "  Device: $DEVICE"
echo "=========================================="
echo ""

# -----------------------------------------------
# 1. Heat (Nordbyen)
# -----------------------------------------------
echo "[1/3] Heat (Nordbyen) ..."
python3 "$SCRIPT_DIR/interpret_tsnamlss.py" \
    --csv_path            "$PROJECT_ROOT/processing/nordbyen_processing/nordbyen_features_engineered.csv" \
    --ckpt                "$PROJECT_ROOT/models/nordbyen_heat/NAMLSS.pt" \
    --preprocessing_state "$PROJECT_ROOT/models/nordbyen_heat/NAMLSS_preprocessing_state.pkl" \
    --test_start_str 2020-01-01 \
    --test_end_str   2020-11-14 \
    $DEVICE \
    --out_dir "$PROJECT_ROOT/interp_out/nordbyen_NAMLSS" \
    --do_effects \
    --plot_full_dataset

echo ""
echo "  ✓ Saved to: interp_out/nordbyen_NAMLSS/"
echo ""

# -----------------------------------------------
# 2. Water (Centrum)
# -----------------------------------------------
echo "[2/3] Water (Centrum) ..."
python3 "$SCRIPT_DIR/interpret_tsnamlss.py" \
    --csv_path            "$PROJECT_ROOT/processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv" \
    --ckpt                "$PROJECT_ROOT/models/water_centrum/NAMLSS.pt" \
    --preprocessing_state "$PROJECT_ROOT/models/water_centrum/NAMLSS_preprocessing_state.pkl" \
    --test_start_str 2020-05-07 \
    --test_end_str   2020-11-14 \
    $DEVICE \
    --out_dir "$PROJECT_ROOT/interp_out/centrum_NAMLSS" \
    --do_effects \
    --plot_full_dataset

echo ""
echo "  ✓ Saved to: interp_out/centrum_NAMLSS/"
echo ""

# -----------------------------------------------
# 3. Water (Tommerby)
# -----------------------------------------------
echo "[3/3] Water (Tommerby) ..."
python3 "$SCRIPT_DIR/interpret_tsnamlss.py" \
    --csv_path            "$PROJECT_ROOT/processing/tommerby_processing/tommerby_features_engineered_from_2018-04-01.csv" \
    --ckpt                "$PROJECT_ROOT/models/water_tommerby/NAMLSS.pt" \
    --preprocessing_state "$PROJECT_ROOT/models/water_tommerby/NAMLSS_preprocessing_state.pkl" \
    --test_start_str 2020-05-07 \
    --test_end_str   2020-11-14 \
    $DEVICE \
    --out_dir "$PROJECT_ROOT/interp_out/tommerby_NAMLSS" \
    --do_effects \
    --plot_full_dataset

echo ""
echo "  ✓ Saved to: interp_out/tommerby_NAMLSS/"
echo ""

echo "=========================================="
echo "  All done! Interpretation artifacts in:"
echo "    $PROJECT_ROOT/interp_out/"
echo "=========================================="
