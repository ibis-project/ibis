#!/bin/bash -e
# Set up conda environment for Ibis in GitHub Actions
# The base environment of the provided conda is used
# This script only installs the base dependencies.
# Dependencies for the backends need to be installed separately.

PYTHON_VERSION="${1:-3.7}"
BACKENDS="$2"

echo "PYTHON_VERSION: $PYTHON_VERSION"
echo "BACKENDS: $BACKENDS"

if [[ -n "$CONDA" ]]; then
    # Add conda to Path
    OS_NAME=$(uname)
    case $OS_NAME in
        Linux)
            CONDA_PATH="$CONDA/bin"
            ;;
        MINGW*)
            # Windows
            CONDA_POSIX=$(cygpath -u "$CONDA")
            CONDA_PATH="$CONDA_POSIX:$CONDA_POSIX/Scripts:$CONDA_POSIX/Library:$CONDA_POSIX/Library/bin:$CONDA_POSIX/Library/mingw-w64/bin"
            ;;
        *)
            echo "$OS_NAME not supported."
            exit 1
    esac
    PATH=${CONDA_PATH}:${PATH}
    # Prepend conda path to system path for the subsequent GitHub Actions
    echo "${CONDA_PATH}" >> $GITHUB_PATH
else
    echo "Running without adding conda to PATH."
fi

conda update -n base -c anaconda --all --yes conda
conda install -n base -c anaconda --yes  python=${PYTHON_VERSION}
conda env update -n base --file=environment.yml
python -m pip install -e .

if [[ -n "$BACKENDS" ]]; then
    python ci/datamgr.py download
    for BACKEND in $BACKENDS; do
        # For the oldest python version supported (currently 3.7) we first try to
        # install the minimum supported dependencies `ci/deps/$BACKEND-min.yml`.
        # If the file does not exist then we install the normal dependencies
        # (if there are dependencies). For other python versions we simply install
        # the normal dependencies if they exist.
        if [[ $PYTHON_VERSION == "3.7" && -f "ci/deps/$BACKEND-min.yml" ]]; then
            conda install -n base -c conda-forge --file="ci/deps/$BACKEND-min.yml"
        else
            if [[ -f "ci/deps/$BACKEND.yml" ]]; then
                conda install -n base -c conda-forge --file="ci/deps/$BACKEND.yml"
            fi
        fi

        # TODO load impala data in the same way as the rest of the backends
        if [[ "$BACKEND" == "impala" ]]; then
            python ci/impalamgr.py load --data
        else
            python ci/datamgr.py $BACKEND
        fi
    done
fi
