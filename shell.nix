{ python ? "3.10" }:
let
  pkgs = import ./nix;

  devDeps = with pkgs; [
    cacert
    cachix
    commitlint
    curl
    git
    niv
    nix-linter
    nixpkgs-fmt
    poetry
    prettierTOML
    shellcheck
    shfmt
  ];

  impalaUdfDeps = with pkgs; [
    boost
    clang_12
    cmake
    ninja
  ];

  backendTestDeps = [ pkgs.docker-compose_2 ];
  vizDeps = [ pkgs.graphviz-nox ];
  pysparkDeps = [ pkgs.openjdk11 ];
  docDeps = [ pkgs.pandoc ];

  # postgresql is the client, not the server
  postgresDeps = [ pkgs.postgresql ];
  geospatialDeps = with pkgs; [ gdal proj ];

  sqliteDeps = [ pkgs.sqlite-interactive ];

  libraryDevDeps = impalaUdfDeps
    ++ backendTestDeps
    ++ vizDeps
    ++ pysparkDeps
    ++ docDeps
    ++ geospatialDeps
    ++ postgresDeps
    ++ sqliteDeps;

  pythonShortVersion = builtins.replaceStrings [ "." ] [ "" ] python;
  pythonEnv = pkgs."ibisDevEnv${pythonShortVersion}";
in
pkgs.mkShell {
  name = "ibis${pythonShortVersion}";

  shellHook = ''
    data_dir="$PWD/ci/ibis-testing-data"
    mkdir -p "$data_dir"
    chmod u+rwx "$data_dir"
    cp -rf ${pkgs.ibisTestingData}/* "$data_dir"
    chmod --recursive u+rw "$data_dir"

    export IBIS_TEST_DATA_DIRECTORY="$data_dir"
  '';

  buildInputs = devDeps ++ libraryDevDeps ++ [
    pythonEnv
    (pythonEnv.python.pkgs.toPythonApplication pkgs.pre-commit)
  ];

  PYTHONPATH = builtins.toPath ./.;
}
