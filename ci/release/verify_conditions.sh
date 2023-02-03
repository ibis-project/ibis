#!/usr/bin/env bash

set -euo pipefail

dry_run="${1:-false}"

# verify pyproject.toml
nix develop '.#release' -c poetry check

# verify that the lock file matches pyproject.toml
#
# the lock file might not be the most fresh, but that's okay: it need only be
# consistent with pyproject.toml
nix develop '.#release' -c poetry lock --check

# verify that we have a token available to push to pypi using set -u
if [ "${dry_run}" = "false" ]; then
  : "${POETRY_PYPI_TOKEN_PYPI}"
fi
