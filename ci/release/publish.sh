#!/usr/bin/env bash

set -euo pipefail

nix develop -c poetry publish
