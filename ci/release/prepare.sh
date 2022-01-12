#!/usr/bin/env nix-shell
#!nix-shell -p poetry nix -i bash
# shellcheck shell=bash

set -euo pipefail

# set version
poetry version "$1"

./dev/poetry2setup -o setup.py

# build artifacts
poetry build
