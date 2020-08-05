#!/bin/bash -e

PYTHON_VERSION="$1"
CONDA_PATH="/usr/share/miniconda/bin"
PATH=${CONDA_PATH}:${PATH}

echo "::add-path::${CONDA_PATH}"

conda update -n base -c anaconda conda
conda install -n base -c anaconda python=${PYTHON_VERSION}
conda env update -n base --file=ci/requirements-dev-3.7-main.yml
python -m pip install -e .
