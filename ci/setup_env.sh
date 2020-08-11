#!/bin/bash -e
# Set up conda environment for Ibis in GitHub Actions
# The base environment of the provided conda is used
# This script only installs the base dependencies.
# Dependencies for the backends need to be installed separately.

# FIXME trying to find conda in windows, remove later
set -x
cd /
find . -name "conda*"


PYTHON_VERSION="${1:-3.7}"
BACKENDS="$2"
CONDA_PATH="/usr/share/miniconda/bin"
PATH=${CONDA_PATH}:${PATH}

echo "PYTHON_VERSION: $PYTHON_VERSION"
echo "BACKENDS: $BACKENDS"
echo "::add-path::${CONDA_PATH}"

conda update -n base -c anaconda --all --yes conda
conda install -n base -c anaconda --yes  python=${PYTHON_VERSION}
conda env update -n base --file=environment.yml
python -m pip install -e .

if [[ -n "$BACKENDS" ]]; then
    python ci/datamgr.py download
    for BACKEND in $BACKENDS; do
        if [[ -f "ci/deps/$BACKEND.yml" ]]; then
            conda install -n base -c conda-forge --file="ci/deps/$BACKEND.yml"
        fi

        # TODO load impala data in the same way as the rest of the backends
        if [[ "$BACKEND" == "impala" ]]; then
            python ci/impalamgr.py load --data
        else
            python ci/datamgr.py $BACKEND
        fi
    done
fi
