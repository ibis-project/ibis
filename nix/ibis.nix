{ poetry2nix
, python3
, lib
, gitignoreSource
, graphviz-nox
, sqlite
, ibisTestingData
}:
let
  backends = [ "datafusion" "duckdb" "pandas" "polars" "sqlite" ]
    # dask version has a show-stopping bug for Python >=3.11
    ++ lib.optionals (python3.pythonOlder "3.11") [ "dask" ];
  markers = lib.concatStringsSep " or " (backends ++ [ "core" ]);
in
poetry2nix.mkPoetryApplication rec {
  python = python3;
  groups = [ ];
  checkGroups = [ "test" ];
  projectDir = gitignoreSource ../.;
  src = gitignoreSource ../.;
  extras = backends ++ [ "decompiler" "visualization" ];
  overrides = [
    (import ../poetry-overrides.nix)
    poetry2nix.defaultPoetryOverrides
  ];
  preferWheels = true;
  __darwinAllowLocalNetworking = true;

  POETRY_DYNAMIC_VERSIONING_BYPASS = "1";

  nativeCheckInputs = [ graphviz-nox sqlite ];

  preCheck = ''
    set -euo pipefail

    HOME="$(mktemp -d)"
    export HOME

    ln -s "${ibisTestingData}" $PWD/ci/ibis-testing-data
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
