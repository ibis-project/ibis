#!/usr/bin/env bash

set -euo pipefail

curdir="$PWD"
worktree="$(mktemp -d)"
branch="$(basename "$worktree")"

nix develop -c git worktree add "$worktree"

function cleanup() {
  cd "$curdir" || exit 1
  nix develop -c git worktree remove --force "$worktree"
  nix develop -c git worktree prune
  nix develop -c git branch -D "$branch"
}

trap cleanup EXIT ERR

cd "$worktree" || exit 1

nix develop -c node <<< 'console.log(JSON.stringify(require("./.releaserc.js")))' |
  jq '.plugins |= [.[] | select(.[0] != "@semantic-release/github")]' > .releaserc.json

nix develop -c git rm .releaserc.js

nix develop -c git add .releaserc.json

nix develop -c git commit -m 'test: semantic-release dry run' --no-verify --no-gpg-sign

unset GITHUB_ACTIONS

nix develop -c npx --yes \
  -p semantic-release \
  -p "@semantic-release/commit-analyzer" \
  -p "@semantic-release/release-notes-generator" \
  -p "@semantic-release/changelog" \
  -p "@semantic-release/exec" \
  -p "@semantic-release/git" \
  -p "@google/semantic-release-replace-plugin" \
  -p "conventional-changelog-conventionalcommits" \
  semantic-release \
  --ci \
  --dry-run \
  --branches "$branch" \
  --repository-url "file://$PWD"
