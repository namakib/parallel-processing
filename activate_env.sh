#!/bin/bash

# Activation script for ParallelProcessing virtual environment
# Usage: source activate_env.sh

echo "🚀 Activating ParallelProcessing virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📍 Current Python: $(which python)"
echo "📦 Installed packages:"
pip list | grep -E "(numpy|matplotlib|numba)"

echo ""
echo "🎯 Ready to run assignments:"
echo "  cd Assignment1 && python Assignment1.py"
echo "  cd Assignment2 && python Assignment2.py"  
echo "  cd Assignment3 && python Assignment3.py"
echo ""
echo "💡 To deactivate: deactivate"
