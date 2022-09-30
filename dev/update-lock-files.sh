#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=channel:nixos-unstable --pure -p poetry nix -i bash
# shellcheck shell=bash
set -euo pipefail

export PYTHONHASHSEED=0

TOP="${1:-$(dirname "$(dirname "$(readlink -f "$0")")")}"

pushd "${TOP}" > /dev/null || exit 1
poetry lock --no-update
poetry export --dev --without-hashes --no-ansi --extras all > "${TOP}/requirements.txt"
popd > /dev/null || exit 1
