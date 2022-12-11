{ poetry2nix
, python3
, lib
, gitignoreSource
, graphviz-nox
, sqlite
, rsync
, ibisTestingData
}:
let
  backends = [ "dask" "datafusion" "duckdb" "pandas" "polars" "sqlite" ];
in
poetry2nix.mkPoetryApplication rec {
  python = python3;
  groups = [ ];
  checkGroups = [ "test" ];
  projectDir = gitignoreSource ../.;
  src = gitignoreSource ../.;
  extras = backends;
  overrides = [
    (import ../poetry-overrides.nix)
    poetry2nix.defaultPoetryOverrides
  ];
  preferWheels = true;

  buildInputs = [ graphviz-nox sqlite ];
  checkInputs = buildInputs;

  preCheck = ''
    set -euo pipefail

    export IBIS_TEST_DATA_DIRECTORY="$PWD/ci/ibis-testing-data"

    ${rsync}/bin/rsync \
      --chmod=Du+rwx,Fu+rw --archive --delete \
      "${ibisTestingData}/" \
      "$IBIS_TEST_DATA_DIRECTORY"
  '';

  checkPhase = ''
    set -euo pipefail

    runHook preCheck

    pytest \
      --numprocesses "$NIX_BUILD_CORES" \
      --dist loadgroup \
      -m '${lib.concatStringsSep " or " backends} or core'

    runHook postCheck
  '';

  doCheck = true;

  pythonImportsCheck = [ "ibis" ] ++ (map (backend: "ibis.backends.${backend}") backends);
}
