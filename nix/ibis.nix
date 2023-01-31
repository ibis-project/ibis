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
  markers = lib.concatStringsSep " or " (backends ++ [ "core" ]);
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
  nativeCheckInputs = checkInputs;

  preCheck = ''
    set -euo pipefail

    HOME="$(mktemp -d)"
    export HOME

    ${rsync}/bin/rsync \
      --chmod=Du+rwx,Fu+rw --archive --delete \
      "${ibisTestingData}/" $PWD/ci/ibis-testing-data
  '';

  checkPhase = ''
    set -euo pipefail

    runHook preCheck

    # the sqlite-on-duckdb tests try to download the sqlite_scanner extension
    # but network usage is not allowed in the sandbox
    pytest -m '${markers}' --numprocesses "$NIX_BUILD_CORES" --dist loadgroup \
      --deselect=ibis/backends/duckdb/tests/test_register.py::test_{read,register}_sqlite \

    runHook postCheck
  '';

  doCheck = true;

  pythonImportsCheck = [ "ibis" ] ++ (map (backend: "ibis.backends.${backend}") backends);
}
