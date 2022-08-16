#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=channel:nixos-unstable-small --pure -p dyff git poetry yj -i bash
# shellcheck shell=bash

set -euo pipefail

old="$(mktemp --suffix=".yaml")"
new="$(mktemp --suffix=".yaml")"

# verify that the lock file is up to date
#
# go through the rigamarole of yj and dyff because poetry is sensitive to
# PYTHONHASHSEED
yj -ty < poetry.lock > "$old"
PYTHONHASHSEED=42 poetry lock --no-update
yj -ty < poetry.lock > "$new"

if ! dyff between "$old" "$new" --ignore-order-changes --omit-header --set-exit-code; then
  git checkout poetry.lock
  exit 1
fi
