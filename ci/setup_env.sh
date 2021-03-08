#!/bin/bash -e
# Set up conda environment for Ibis in GitHub Actions
# The base environment of the provided conda is used
# This script only installs the base dependencies.
# Dependencies for the backends need to be installed separately.

PYTHON_VERSION="${1:-3.7}"
BACKENDS="$2"
LOAD_TEST_DATA="${3:-true}"

echo "PYTHON_VERSION: $PYTHON_VERSION"
echo "BACKENDS: $BACKENDS"
echo "LOAD_TEST_DATA: $LOAD_TEST_DATA"

# Install micromamba
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc

# Install base environment
sed "s/dependencies:/dependencies:\n  - python=${PYTHON_VERSION}\n/" environment.yml
micromamba install --file=environment.yml
python -m pip install -e .

if [[ -n "$BACKENDS" ]]; then
    if [[ $LOAD_TEST_DATA == "true" ]]; then
        python ci/datamgr.py download
    fi

    for BACKEND in $BACKENDS; do
        # For the oldest python version supported (currently 3.7) we first try to
        # install the minimum supported dependencies `ci/deps/$BACKEND-min.yml`.
        # If the file does not exist then we install the normal dependencies
        # (if there are dependencies). For other python versions we simply install
        # the normal dependencies if they exist.
        if [[ $PYTHON_VERSION == "3.7" && -f "ci/deps/$BACKEND-min.yml" ]]; then
            micromamba install --file="ci/deps/$BACKEND-min.yml"
        else
            if [[ -f "ci/deps/$BACKEND.yml" ]]; then
                micromamba install --file="ci/deps/$BACKEND.yml"
            fi
        fi

        if [[ $LOAD_TEST_DATA == "true" ]]; then
            # TODO load impala data in the same way as the rest of the backends
            if [[ "$BACKEND" == "impala" ]]; then
                python ci/impalamgr.py load --data
            else
                python ci/datamgr.py $BACKEND
            fi
        fi
    done
fi

conda list
