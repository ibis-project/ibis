#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=channel:nixos-unstable-small --pure --keep POETRY_PYPI_TOKEN_PYPI -p poetry git jd-diff-patch yj -i bash
# shellcheck shell=bash

set -euo pipefail

dry_run="${1}"

# verify pyproject.toml
poetry check

# verify that the lock file is up to date
#
# go through the rigamarole of yj and jd because poetry is sensitive to
# PYTHONHASHSEED
bash ./dev/lockfile_diff.sh

git status --short poetry.lock
git checkout poetry.lock

# verify that we have a token available to push to pypi using set -u
if [ "${dry_run}" = "false" ]; then
  : "${POETRY_PYPI_TOKEN_PYPI}"
fi
