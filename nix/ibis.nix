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
  __darwinAllowLocalNetworking = true;

  POETRY_DYNAMIC_VERSIONING_BYPASS = "1";

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

    pytest -m '${markers}' --numprocesses "$NIX_BUILD_CORES" --dist loadgroup

    runHook postCheck
  '';

  doCheck = true;

  pythonImportsCheck = [ "ibis" ] ++ map (backend: "ibis.backends.${backend}") backends;
}
