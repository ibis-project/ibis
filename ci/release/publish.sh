#!/usr/bin/env bash

set -euo pipefail

nix develop '.#release' -c uv publish
