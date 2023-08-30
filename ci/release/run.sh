#!/usr/bin/env bash

set -euo pipefail

nix develop '.#release' -c npx --yes \
  -p semantic-release \
  -p "@semantic-release/commit-analyzer" \
  -p "@semantic-release/release-notes-generator" \
  -p "@semantic-release/changelog" \
  -p "@semantic-release/github" \
  -p "@semantic-release/exec" \
  -p "@semantic-release/git" \
  -p "semantic-release-replace-plugin@1.2.0" \
  -p "conventional-changelog-conventionalcommits@6.1.0" \
  semantic-release --ci
