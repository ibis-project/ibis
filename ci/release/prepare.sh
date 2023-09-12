#!/usr/bin/env bash

set -euo pipefail

version="${1}"

# set version
nix develop '.#release' -c poetry version "$version"

# build artifacts
nix develop '.#release' -c poetry build

# ensure that the built wheel has the correct version number
nix develop '.#release' -c unzip -p "dist/ibis_framework-${version}-py3-none-any.whl" ibis/__init__.py | \
  nix develop '.#release' -c grep -q "__version__ = \"$version\""
