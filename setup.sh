#!/usr/bin/env bash
# ==============================================================================
# Script di installazione automatizzata per Debian / Ubuntu
# ==============================================================================
set -e  # Interrompe lo script in caso di errore

echo "======================================================================"
echo " 1. Installing system dependencies (APT)"
echo "======================================================================"
sudo apt update
sudo apt install -y curl git build-essential libasound2-dev python3-dev

echo ""
echo "======================================================================"
echo " 2. Installation / Verification of Miniconda"
echo "======================================================================"
CONDA_DIR="$HOME/miniconda3"

if [ ! -d "$CONDA_DIR" ]; then
    echo "Miniconda not found. Starting download and installation..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "Miniconda successfully installed in $CONDA_DIR"
else
    echo "Miniconda is already installed in $CONDA_DIR"
fi

# Carica l'ambiente Conda nella sessione di questo script
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo ""
echo "======================================================================"
echo " 3. Initialization / Update Conda Environment"
echo "======================================================================"
if [ ! -f "environment.yml" ]; then
    echo "ERROR: File environment.yml not found in the current directory!"
    exit 1
fi

ENV_NAME=$(grep "^name:" environment.yml | head -n 1 | awk '{print $2}')

if [ -z "$ENV_NAME" ]; then
    ENV_NAME="BAT"
    echo "ERROR: Unable to detect the environment name from environment.yml, using BAT as default"
    exit 1
fi

if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "The environment '$ENV_NAME' already exists. Updating..."
    conda env update -f environment.yml --prune
else
    echo "Creating new Conda environment '$ENV_NAME'..."
    conda env create -f environment.yml
fi

echo ""
echo "======================================================================"
echo " 4. Installation of Current Package"
echo "======================================================================"
conda activate "$ENV_NAME"

if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing the local package in editable mode (pip install -e .)..."
    pip install -e .
else
    echo "No setup.py or pyproject.toml found. Skipping local installation."
fi

echo ""
echo "======================================================================"
echo " Installation completed successfully  !"
echo "======================================================================"
