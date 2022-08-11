#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=channel:nixos-unstable-small --pure -p poetry jd-diff-patch yj -i bash
# shellcheck shell=bash

set -euo pipefail

export PYTHONHASHSEED=42

old="$(mktemp --suffix=".json")"
new="$(mktemp --suffix=".json")"

# verify that the lock file is up to date
#
# go through the rigamarole of yj and jd because poetry is sensitive to
# PYTHONHASHSEED
yj -tj < poetry.lock > "$old"
poetry lock --no-update
yj -tj < poetry.lock > "$new"

# -set treats arrays as unordered sequences with unique elements
# for the purposes of comparison
jd -set "$old" "$new"
