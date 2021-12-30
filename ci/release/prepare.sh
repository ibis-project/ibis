#!/usr/bin/env nix-shell
#!nix-shell --pure -p poetry -i bash
# shellcheck shell=bash

set -euo pipefail

# set version
poetry version "$1"

# build artifacts
poetry build
