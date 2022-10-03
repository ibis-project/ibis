{ python ? "3.10"
, doCheck ? true
, backends ? [
    "dask"
    "datafusion"
    "duckdb"
    "pandas"
    "sqlite"
  ]
}:
let
  pkgs = import ./nix { };
  drv = { poetry2nix, python, lib }: poetry2nix.mkPoetryApplication rec {
    inherit python;

    groups = [ ];
    checkGroups = lib.optionals doCheck [ "test" ];
    projectDir = ./.;
    src = pkgs.gitignoreSource ./.;

    overrides = pkgs.poetry2nix.overrides.withDefaults (
      import ./poetry-overrides.nix
    );

    buildInputs = with pkgs; [ gdal graphviz-nox proj sqlite ];
    checkInputs = buildInputs;

    preCheck = ''
      set -euo pipefail

      export IBIS_TEST_DATA_DIRECTORY="$PWD/ci/ibis-testing-data"

      ${pkgs.rsync}/bin/rsync \
        --chmod=Du+rwx,Fu+rw --archive --delete \
        "${pkgs.ibisTestingData}/" \
        "$IBIS_TEST_DATA_DIRECTORY"
    '';

    checkPhase = ''
      set -euo pipefail

      runHook preCheck

      pytest --numprocesses auto --dist loadgroup -m '${lib.concatStringsSep " or " backends} or core'

      runHook postCheck
    '';

    inherit doCheck;

    pythonImportsCheck = [ "ibis" ] ++ (map (backend: "ibis.backends.${backend}") backends);
  };
in
pkgs.callPackage drv {
  python = pkgs."python${builtins.replaceStrings [ "." ] [ "" ] python}";
}
