#!/bin/bash
# Quick setup script for the differential privacy repository

set -e  # Exit on error

echo "=========================================="
echo "Differential Privacy Repository Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Already in a virtual environment: $VIRTUAL_ENV"
else
    echo "⚠ Not in a virtual environment"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
    echo "To activate it, run:"
    echo "  source venv/bin/activate  # On Linux/Mac"
    echo "  venv\\Scripts\\activate     # On Windows"
    echo ""
    read -p "Press Enter to continue (make sure to activate venv first)..."
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p models checkpoints logs data

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test the installation:"
echo "   python train.py --help"
echo ""
echo "2. Train a simple model:"
echo "   python train.py --model autoencoder --epochs 5"
echo ""
echo "3. Read the documentation:"
echo "   cat README.md"
echo "   cat IMPROVEMENTS.md"
echo ""
echo "Happy coding!"
