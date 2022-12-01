#!/usr/bin/env bash
set -euo pipefail

top="$PWD"
clone="${1}"
tag="$(nix develop -c yj -tj < "${top}/poetry.lock" | nix develop -c jq -rcM '.package[] | select(.name == "datafusion") | .version')"

nix develop -c git -C "${clone}" fetch
nix develop -c git -C "${clone}" checkout .
nix develop -c git -C "${clone}" checkout "${tag}"

# use thin lto to dramatically speed up builds
Nix develop -c sed -i \
  -e '/codegen-units = 1/d' \
  -e 's/lto = true/lto = "thin"/g' \
  "${clone}/Cargo.toml"

pushd "${clone}"
nix develop -c cargo generate-lockfile
popd

mkdir -p "${top}/nix/patches"

nix develop -c git -C "${clone}" diff > "${top}/nix/patches/datafusion.patch"
