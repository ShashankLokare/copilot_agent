#!/bin/bash
#
# Install NIFTY50 Data Collection & ML Training Dependencies
#
# This script installs all required packages for:
# - Collecting 15 years of NIFTY50 data
# - Preparing data for ML training
# - Training prediction models
#

echo "================================================"
echo "Installing NIFTY50 ML Pipeline Dependencies"
echo "================================================"

# Core data science
echo ""
echo "[1/6] Installing core data science packages..."
pip install pandas numpy scipy scikit-learn -q

# Machine Learning
echo "[2/6] Installing machine learning libraries..."
pip install xgboost lightgbm -q

# Data Collection
echo "[3/6] Installing data collection tools..."
pip install yfinance -q
pip install nsepy -q 2>/dev/null || echo "    (nsepy optional, skipped)"

# Visualization
echo "[4/6] Installing visualization libraries..."
pip install matplotlib seaborn plotly -q

# Utilities
echo "[5/6] Installing utility packages..."
pip install python-dateutil pytz requests -q

# Development
echo "[6/6] Installing development tools..."
pip install jupyter ipython pytest -q

echo ""
echo "================================================"
echo "✓ Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Collect data: python scripts/collect_nifty50_15y_data.py"
echo "2. Prepare data: python scripts/prepare_nifty50_data.py"
echo "3. Train models: python scripts/train_nifty50_models.py"
echo ""
