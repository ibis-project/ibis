#!/usr/bin/env bash

set -euo pipefail

nix develop '.#release' -c poetry publish
