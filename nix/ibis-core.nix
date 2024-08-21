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
mkApp { }
