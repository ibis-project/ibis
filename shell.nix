{ python ? "3.10" }:
let
  pkgs = import ./nix;

  pythonEnv = pkgs."ibisDevEnv${pythonShortVersion}";

  devDeps = with pkgs; [
    # terminal markdown rendering
    glow
    # json diffing, executable is jd
    jd-diff-patch
    # used in the justfile
    jq
    yj
    # linting
    commitlint
    lychee
    # packaging
    niv
    pythonEnv.pkgs.poetry
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

  updateLockFiles = pkgs.writeShellApplication {
    name = "update-lock-files";
    runtimeInputs = [ pythonEnv.pkgs.poetry ];
    text = ''
      export PYTHONHASHSEED=42
      poetry lock --no-update
      poetry export --dev --without-hashes --no-ansi --extras all > requirements.txt
      ./dev/poetry2setup -o setup.py
    '';
  };
in
pkgs.mkShell {
  name = "ibis${pythonShortVersion}";

  shellHook = ''
    export IBIS_TEST_DATA_DIRECTORY="$PWD/ci/ibis-testing-data"

    rsync \
      --chmod=Du+rwx,Fu+rw --archive --delete \
      "${pkgs.ibisTestingData}/" \
      "$IBIS_TEST_DATA_DIRECTORY"

    export TEMPDIR
    TEMPDIR="$(python -c 'import tempfile; print(tempfile.gettempdir())')"
  '';

  buildInputs = devDeps ++ libraryDevDeps ++ [
    pythonEnv
    updateLockFiles
  ] ++ pkgs.preCommitShell.buildInputs ++ (with pkgs; [
    changelog
    mic
    rsync
  ]);

  PYTHONPATH = builtins.toPath ./.;
  PGPASSWORD = "postgres";
  MYSQL_PWD = "ibis";
}
