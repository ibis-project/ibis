#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=0

TOP="${1:-$(dirname "$(dirname "$(readlink -f "$0")")")}"

pushd "${TOP}" > /dev/null || exit 1
nix run 'nixpkgs#poetry' -- lock --no-update
nix run 'nixpkgs#poetry' -- export --with dev --with test --with docs --without-hashes --no-ansi > "${TOP}/requirements.txt"
popd > /dev/null || exit 1
