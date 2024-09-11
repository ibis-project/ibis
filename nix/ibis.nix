{ poetry2nix
, python3
, lib
, gitignoreSource
, graphviz-nox
, sqlite
, ibisTestingData
}:
# pyspark could be added here, but it doesn't handle parallel test execution
# well and serially it takes on the order of 7-8 minutes to execute serially
let
  extras = [ "decompiler" "visualization" ];
  backends = [ "datafusion" "duckdb" "pandas" "polars" "sqlite" ]
    # dask version has a show-stopping bug for Python >=3.11
    ++ lib.optionals (python3.pythonOlder "3.11") [ "dask" ];
in
poetry2nix.mkPoetryApplication {
  python = python3;
  groups = [ ];
  checkGroups = [ "test" ];
  projectDir = gitignoreSource ../.;
  src = gitignoreSource ../.;
  extras = backends ++ extras;
  overrides = [
    (import ../poetry-overrides.nix)
    poetry2nix.defaultPoetryOverrides
  ];
  preferWheels = true;
  __darwinAllowLocalNetworking = true;

  POETRY_DYNAMIC_VERSIONING_BYPASS = "1";

  nativeCheckInputs = lib.optionals (lib.elem "sqlite" backends) [ sqlite ]
    ++ lib.optionals (lib.elem "visualization" extras) [ graphviz-nox ];

  preCheck = ''
    set -euo pipefail

    HOME="$(mktemp -d)"
    export HOME

    ln -s "${ibisTestingData}" $PWD/ci/ibis-testing-data
  '';

  checkPhase =
    let
      markers = lib.concatStringsSep " or " (backends ++ [ "core" ]);
    in
    ''
      set -euo pipefail

      runHook preCheck

      pytest -m 'not tpcds and (${markers})'

      runHook postCheck
    '';

  doCheck = true;

  pythonImportsCheck = [ "ibis" ] ++ map (backend: "ibis.backends.${backend}") backends;
}
