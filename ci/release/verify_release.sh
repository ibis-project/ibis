#!/usr/bin/env bash

set -euo pipefail

next_version="${1}"
python ci/release/verify_release.py "$next_version" --root "$PWD"
