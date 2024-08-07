{ poetry2nix
, python3
, lib
, gitignoreSource
, graphviz-nox
, sqlite
, ibisTestingData
}:
let
  mkApp = import ./ibis.nix {
    inherit poetry2nix python3 lib gitignoreSource graphviz-nox sqlite ibisTestingData;
  };
in
mkApp {
  extras = [ "decompiler" "visualization" ];
  backends = [ "datafusion" "duckdb" "pandas" "polars" "sqlite" ]
    # dask version has a show-stopping bug for Python >=3.11
    ++ lib.optionals (python3.pythonOlder "3.11") [ "dask" ];
}
