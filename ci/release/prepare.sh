#!/usr/bin/env bash

set -euo pipefail

version="${1}"

# set version
nix develop -c poetry version "$version"

# build artifacts
nix develop -c poetry build

# ensure that the built wheel has the correct version number
nix develop -c unzip -p "dist/ibis_framework-${version}-py3-none-any.whl" ibis/__init__.py | grep -q "__version__ = \"$version\""
