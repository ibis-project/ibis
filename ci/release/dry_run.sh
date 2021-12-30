#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=channel:nixos-unstable-small --pure -p git nodejs nix -i bash
# shellcheck shell=bash

set -euo pipefail

curdir="$PWD"
worktree="$(mktemp -d)"
branch="$(basename "$worktree")"

git worktree add "$worktree"

function cleanup() {
  cd "$curdir" || exit 1
  git worktree remove "$worktree"
  git worktree prune
  git branch -D "$branch"
}

trap cleanup EXIT ERR

cd "$worktree" || exit 1

npx --yes \
  -p semantic-release \
  -p "@semantic-release/commit-analyzer" \
  -p "@semantic-release/release-notes-generator" \
  -p "@semantic-release/changelog" \
  -p "@semantic-release/exec" \
  -p "@semantic-release/git" \
  semantic-release \
  --ci \
  --dry-run \
  --plugins \
  --analyze-commits "@semantic-release/commit-analyzer" \
  --generate-notes "@semantic-release/release-notes-generator" \
  --verify-conditions "@semantic-release/changelog,@semantic-release/exec,@semantic-release/git" \
  --prepare "@semantic-release/changelog,@semantic-release/exec" \
  --branches "$branch" \
  --repository-url "file://$PWD"
