#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=./nix -p git gnused sd rustNightly yj jq -i bash
# shellcheck shell=bash
set -euo pipefail

top="$PWD"
clone="${1}"
tag="py-$(yj -tj < "${top}/poetry.lock" | jq '.package[] | select(.name == "polars") | .version' -rcM)"

git -C "${clone}" fetch
git -C "${clone}" checkout .
git -C "${clone}" checkout "${tag}"

# 1. remove patch dependencies
# 2. use thin lto to dramatically speed up builds
sed -i -e '/\[patch\.crates-io\]/d' -e '/cmake = .*/,+2d' -e '/codegen-units = 1/d' -e 's/lto = "fat"/lto = "thin"/g' "${clone}/py-polars/Cargo.toml"

pushd "${clone}/py-polars"
cargo generate-lockfile
popd

git -C "${clone}" diff | sd 'py-polars/' '' > "${top}/nix/patches/py-polars.patch"
