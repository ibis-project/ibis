#!/usr/bin/env bash

set -euo pipefail

dry_run="${1:-false}"

# verify that the lock file matches pyproject.toml
nix develop '.#release' -c uv lock --locked

# verify that we have a token available to push to pypi using set -u
if [ "${dry_run}" = "false" ]; then
  : "${UV_PUBLISH_TOKEN}"
fi
