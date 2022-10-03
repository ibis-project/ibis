#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=./nix --pure --keep POETRY_PYPI_TOKEN_PYPI -p poetry -i bash
# shellcheck shell=bash

set -euo pipefail

poetry publish
