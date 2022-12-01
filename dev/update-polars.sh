#!/usr/bin/env bash
set -euo pipefail

top="$PWD"
clone="${1}"
tag="py-$(yj -tj < "${top}/poetry.lock" | jq '.package[] | select(.name == "polars") | .version' -rcM)"

nix develop -c git -C "${clone}" fetch
nix develop -c git -C "${clone}" checkout .
nix develop -c git -C "${clone}" checkout "${tag}"

# remove patch dependencies and use thin lto to dramatically speed up builds
nix develop -c sed -i \
  -e '/\[patch\.crates-io\]/d' \
  -e '/cmake = .*/,+2d' \
  -e '/codegen-units = 1/d' \
  -e 's/lto = "fat"/lto = "thin"/g' \
  "${clone}/py-polars/Cargo.toml"

pushd "${clone}/py-polars"
nix develop -c cargo generate-lockfile
popd

mkdir -p "${top}/nix/patches"

nix develop -c git -C "${clone}" diff | nix develop -c sd 'py-polars/' '' > "${top}/nix/patches/py-polars.patch"
