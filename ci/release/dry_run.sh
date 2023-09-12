#!/usr/bin/env bash

set -euo pipefail

curdir="$PWD"
worktree="$(mktemp -d)"
branch="$(basename "$worktree")"

nix develop '.#release' -c git worktree add "$worktree"

function cleanup() {
  cd "$curdir" || exit 1
  nix develop '.#release' -c git worktree remove --force "$worktree"
  nix develop '.#release' -c git worktree prune
  nix develop '.#release' -c git branch -D "$branch"
}

trap cleanup EXIT ERR

cd "$worktree" || exit 1

nix develop '.#release' -c node <<< 'console.log(JSON.stringify(require("./.releaserc.js")))' |
  nix develop '.#release' -c jq '.plugins |= [.[] | select(.[0] != "@semantic-release/github")]' > .releaserc.json

nix develop '.#release' -c git rm .releaserc.js
nix develop '.#release' -c git add .releaserc.json
nix develop '.#release' -c git commit -m 'test: semantic-release dry run' --no-verify --no-gpg-sign

# If this is set then semantic-release will assume the release is running
# against a PR.
#
# Normally this would be fine, except that most of the release process that is
# useful to test is prevented from running, even in dry-run mode, so we `unset`
# this variable here and pass `--dry-run` ourselves
unset GITHUB_ACTIONS

nix develop '.#release' -c npx --yes \
  -p semantic-release \
  -p "@semantic-release/commit-analyzer" \
  -p "@semantic-release/release-notes-generator" \
  -p "@semantic-release/changelog" \
  -p "@semantic-release/exec" \
  -p "@semantic-release/git" \
  -p "semantic-release-replace-plugin@1.2.0" \
  -p "conventional-changelog-conventionalcommits@6.1.0" \
  semantic-release \
  --ci \
  --dry-run \
  --branches "$branch" \
  --repository-url "file://$PWD"
