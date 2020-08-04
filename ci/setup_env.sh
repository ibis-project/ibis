#!/bin/bash -e

BASE_DIR="$(readlink -m $(dirname $0)/..)"

echo "Downloading and installing Miniconda..."
UNAME_OS=$(uname)
if [[ "$UNAME_OS" == 'Linux' ]]; then
    if [[ "$BITS32" == "yes" ]]; then
        CONDA_OS="Linux-x86"
    else
        CONDA_OS="Linux-x86_64"
    fi
elif [[ "$UNAME_OS" == 'Darwin' ]]; then
    CONDA_OS="MacOSX-x86_64"
else
  echo "OS $UNAME_OS not supported"
  exit 1
fi
wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-$CONDA_OS.sh -O $BASE_DIR/miniconda.sh
chmod +x $BASE_DIR/miniconda.sh
$BASE_DIR/miniconda.sh -b -p $BASE_DIR/miniconda3
export PATH=$BASE_DIR/miniconda3/bin:$PATH

echo
echo "Configuring and updating conda..."
conda config --set ssl_verify false
conda config --set quiet true --set always_yes true --set changeps1 false
conda update -n base conda

echo
echo "conda env create -q --file=$BASE_DIR/ci/requirements-dev.yml"
time conda env create -q --file="$BASE_DIR/ci/requirements-dev.yml"

echo "Installing Ibis in the environment..."
python -m pip install --no-build-isolation -e $BASE_DIR

echo
echo "conda list"
conda list
