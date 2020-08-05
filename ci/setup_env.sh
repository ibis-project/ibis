#!/bin/bash -e
# Set up conda environment for Ibis in GitHub Actions
# The base environment of the provided conda is used
# This script only installs the base dependencies.
# Dependencies for the backends need to be installed separately.

PYTHON_VERSION="${1:-3.7}"
CONDA_PATH="/usr/share/miniconda/bin"
PATH=${CONDA_PATH}:${PATH}

echo "::add-path::${CONDA_PATH}"

conda update -n base -c anaconda --all --yes conda
conda install -n base -c anaconda --yes  python=${PYTHON_VERSION}
conda env update -n base --file=environment.yml
python -m pip install -e .
