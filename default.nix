{ python ? "3.9" }:
let
  pkgs = import ./nix;
  drv =
    { poetry2nix
    , python
    , lib
    }:

    let
      backends = [
        "csv"
        "dask"
        "hdf5"
        "pandas"
        "parquet"
        "sqlite"
      ];

      backendsString = lib.concatStringsSep " " backends;
    in
    poetry2nix.mkPoetryApplication {
      inherit python;

      pyproject = ./pyproject.toml;
      poetrylock = ./poetry.lock;
      src = lib.cleanSource ./.;

      overrides = pkgs.poetry2nix.overrides.withDefaults (
        import ./poetry-overrides.nix {
          inherit pkgs;
          inherit (pkgs) lib stdenv;
        }
      );

      preConfigure = ''
        rm -f setup.py
      '';

      buildInputs = with pkgs; [ graphviz ];
      checkInputs = with pkgs; [ graphviz ];

      checkPhase = ''
        runHook preCheck

        tempdir="$(mktemp -d)"

        cp -r ${pkgs.ibisTestingData}/* "$tempdir"

        chmod -R u+rwx "$tempdir"

        ln -s "$tempdir" ci/ibis-testing-data

        for backend in ${backendsString}; do
          python ci/datamgr.py "$backend" &
        done

        wait

        PYTEST_BACKENDS="${backendsString}" \
          pytest --numprocesses auto \
          ibis/tests \
          ibis/backends/tests \
          ibis/backends/{${lib.concatStringsSep "," backends}}/tests

        runHook postCheck
      '';

      pythonImportsCheck = [ "ibis" ];
    };
in
pkgs.callPackage drv {
  python = pkgs."python${builtins.replaceStrings [ "." ] [ "" ] python}";
}
