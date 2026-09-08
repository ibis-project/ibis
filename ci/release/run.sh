#!/usr/bin/env bash

set -euo pipefail

nix develop '.#release' -c npx --yes \
  -p "semantic-release@24" \
  -p "@semantic-release/commit-analyzer@13" \
  -p "@semantic-release/release-notes-generator@14" \
  -p "@semantic-release/changelog@6" \
  -p "@semantic-release/github@11" \
  -p "@semantic-release/exec@7" \
  -p "@semantic-release/git@10" \
  -p "semantic-release-replace-plugin@1" \
  -p "conventional-changelog-conventionalcommits@8" \
  semantic-release --ci
