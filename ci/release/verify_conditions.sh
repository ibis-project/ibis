#!/usr/bin/env bash

set -euo pipefail

# verify that the lock file matches pyproject.toml
nix develop '.#release' -c uv lock --locked
