#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=./nix -p poetry nix -i bash
# shellcheck shell=bash
set -euo pipefail

export PYTHONHASHSEED=0

TOP="${1:-$(dirname "$(dirname "$(readlink -f "$0")")")}"

pushd "${TOP}" > /dev/null || exit 1
poetry lock --no-update --no-cache
poetry export --with dev --with test --with docs --without-hashes --no-ansi > "${TOP}/requirements.txt"
popd > /dev/null || exit 1
