#!/usr/bin/env bash
#
# Merge environment files and update the corresponding conda environment

set -euo pipefail

if [ "$#" -eq 0 ]; then
  >&2 echo "error: must provide at least one backend"
  exit 1
fi

# install conda-merge, don't try to update already installed dependencies
mamba install --freeze-installed --name ibis conda-merge

additional_env_files=()

# pull all files associated with input backends
for backend in "$@"; do
  env_file="ci/deps/${backend}.yml"

  if [ -f "${env_file}" ]; then
    additional_env_files+=("$env_file")
  fi
done

env_yaml="$(mktemp --suffix=.yml)"
conda-merge environment.yml "${additional_env_files[@]}" | tee "$env_yaml"
mamba env update --name ibis --file "$env_yaml"
