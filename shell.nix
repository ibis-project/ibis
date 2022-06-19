{ python ? "3.10" }:
let
  pkgs = import ./nix;

  devDeps = with pkgs; [
    # terminal markdown rendering
    glow
    # used in the justfile
    jq
    yj
    # linting
    commitlint
    lychee
    # packaging
    niv
    poetry
  ];

  impalaUdfDeps = with pkgs; [
    clang_12
    cmake
    ninja
  ];
  backendTestDeps = [ pkgs.docker-compose ];
  vizDeps = [ pkgs.graphviz-nox ];
  duckdbDeps = [ pkgs.duckdb ];
  mysqlDeps = [ pkgs.mariadb-client ];
  pysparkDeps = [ pkgs.openjdk11_headless ];

  postgresDeps = [ pkgs.postgresql ]; # postgres client dependencies
  geospatialDeps = with pkgs; [ gdal_2 proj ];
  sqliteDeps = [ pkgs.sqlite-interactive ];

  libraryDevDeps = impalaUdfDeps
    ++ backendTestDeps
    ++ vizDeps
    ++ pysparkDeps
    ++ geospatialDeps
    ++ postgresDeps
    ++ sqliteDeps
    ++ duckdbDeps
    ++ mysqlDeps;

  pythonShortVersion = builtins.replaceStrings [ "." ] [ "" ] python;
  pythonEnv = pkgs."ibisDevEnv${pythonShortVersion}";
  updateLockFiles = pkgs.writeShellApplication {
    name = "update-lock-files";
    text = ''
      poetry export --dev --without-hashes --no-ansi --extras all > requirements.txt
      poetry lock --no-update
    '';
  };
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

    export TEMPDIR
    TEMPDIR="$(python -c 'import tempfile; print(tempfile.gettempdir())')"
  '';

  buildInputs = devDeps ++ libraryDevDeps ++ [
    pkgs.changelog
    pkgs.mic
    pythonEnv
    updateLockFiles
  ] ++ pkgs.preCommitShell.buildInputs;

  PYTHONPATH = builtins.toPath ./.;
}
