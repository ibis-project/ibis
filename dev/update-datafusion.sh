#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=./nix -p git gnused rustNightly yj jq -i bash
# shellcheck shell=bash
set -euo pipefail

top="$PWD"
clone="${1}"
tag="$(yj -tj < "${top}/poetry.lock" | jq '.package[] | select(.name == "datafusion") | .version' -rcM)"

git -C "${clone}" fetch
git -C "${clone}" checkout .
git -C "${clone}" checkout "${tag}"

# 1. use thin lto to dramatically speed up builds
sed -i -e '/codegen-units = 1/d' -e 's/lto = true/lto = "thin"/g' "${clone}/Cargo.toml"

pushd "${clone}"
cargo generate-lockfile
popd

mkdir -p "${top}/nix/patches"

git -C "${clone}" diff > "${top}/nix/patches/datafusion.patch"
