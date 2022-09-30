#!/usr/bin/env nix-shell
#!nix-shell -p gnugrep unzip poetry nix -i bash
# shellcheck shell=bash

set -euo pipefail

version="${1}"

# set version
poetry version "$version"

# build artifacts
poetry build

# ensure that the built wheel has the correct version number
unzip -p "dist/ibis_framework-${version}-py3-none-any.whl" ibis/__init__.py | grep -q "__version__ = \"$version\""
