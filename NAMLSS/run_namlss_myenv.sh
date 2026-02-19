#!/bin/bash
#
# Wrapper script to run NAMLSS training with myenv conda environment
#
# Usage:
#   ./run_namlss_myenv.sh --dataset nordbyen_heat --n_epochs 5
#   ./run_namlss_myenv.sh --dataset water_centrum --n_epochs 30 --use_cqr
#

# Load conda module (HPC environment)
echo "🔄 Loading conda module..."
module load python/3.12-conda

# Check if module load was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to load python/3.12-conda module"
    echo "   Please ensure the module is available: module avail conda"
    exit 1
fi

# Initialize conda for this shell (required for conda activate to work in scripts)
echo "🔄 Initializing conda..."
eval "$(conda shell.bash hook)"

# Activate myenv
echo "🔄 Activating myenv conda environment..."
conda activate myenv

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate myenv"
    echo "   Please ensure myenv exists: conda env list"
    exit 1
fi

# Verify Python is from myenv
PYTHON_PATH=$(which python)
echo "✅ Using Python from: $PYTHON_PATH"
python --version
echo ""

# Double check we have numpy
python -c "import numpy; print(f'✅ NumPy version: {numpy.__version__}')" 2>/dev/null || echo "⚠️  NumPy check failed"
echo ""

# Set PYTHONPATH to include project root (one level up from NAMLSS directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 PYTHONPATH set"
echo ""

# Run the NAMLSS training script with all provided arguments
# Use 'python' (not 'python3') to ensure we use the conda environment's Python
python run_namlss_local.py "$@"

# Capture exit code
EXIT_CODE=$?

# Deactivate environment
conda deactivate

# Exit with the same code as the Python script
exit $EXIT_CODE
