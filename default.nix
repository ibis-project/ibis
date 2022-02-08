{ python ? "3.9"
, doCheck ? true
}:
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
        "datafusion"
        "hdf5"
        "pandas"
        "sqlite"
      ];

      backendsString = lib.concatStringsSep " " backends;
    in
    poetry2nix.mkPoetryApplication {
      inherit python;

      projectDir = ./.;
      src = pkgs.gitignoreSource ./.;

      overrides = pkgs.poetry2nix.overrides.withDefaults (
        import ./poetry-overrides.nix {
          inherit pkgs;
          inherit (pkgs) lib stdenv;
        }
      );

      preConfigure = ''
        rm -f setup.py
      '';

      buildInputs = with pkgs; [ graphviz-nox ];
      checkInputs = with pkgs; [ graphviz-nox ];

      preCheck = ''
        export PYTEST_BACKENDS="${backendsString}"
      '';

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

        pytest --numprocesses auto \
          ibis/tests \
          ibis/backends/tests \
          ibis/backends/{${lib.concatStringsSep "," backends}}/tests

        runHook postCheck
      '';

      inherit doCheck;

      pythonImportsCheck = [ "ibis" ] ++ (map (backend: "ibis.backends.${backend}") backends);
    };
in
pkgs.callPackage drv {
  python = pkgs."python${builtins.replaceStrings [ "." ] [ "" ] python}";
}
