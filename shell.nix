{ python ? "3.10" }:
let
  pkgs = import ./nix { };

  pythonEnv = pkgs."ibisDevEnv${pythonShortVersion}";

  devDeps = with pkgs; [
    # terminal markdown rendering
    glow
    # yaml diffing for poetry.lock via yj -ty ("t"OML to "y"AML)
    dyff
    # used in the justfile
    jq
    yj
    # linting
    commitlint
    lychee
    # packaging
    niv
  ];

  impalaUdfDeps = with pkgs; [
    clang_12
    cmake
    ninja
  ];
  snowflakeDeps = [ pkgs.openssl ];
  backendTestDeps = [ pkgs.docker-compose ];
  vizDeps = [ pkgs.graphviz-nox ];
  duckdbDeps = [ pkgs.duckdb ];
  mysqlDeps = [ pkgs.mariadb-client ];
  pysparkDeps = [ pkgs.openjdk11_headless ];

  postgresDeps = [ pkgs.postgresql ];
  geospatialDeps = [ pkgs.gdal_2 pkgs.proj ];
  sqliteDeps = [ pkgs.sqlite-interactive ];

  libraryDevDeps = impalaUdfDeps
    ++ backendTestDeps
    ++ vizDeps
    ++ pysparkDeps
    ++ geospatialDeps
    ++ postgresDeps
    ++ sqliteDeps
    ++ duckdbDeps
    ++ mysqlDeps
    ++ snowflakeDeps;

  pythonShortVersion = builtins.replaceStrings [ "." ] [ "" ] python;

  updateLockFiles = pkgs.writeShellApplication {
    name = "update-lock-files";
    text = ''
      ${./dev/update-lock-files.sh} "$PWD"
    '';
  };
in
pkgs.mkShell {
  name = "ibis${pythonShortVersion}";

  shellHook = ''
    export IBIS_TEST_DATA_DIRECTORY="$PWD/ci/ibis-testing-data"

    ${pkgs.rsync}/bin/rsync \
      --chmod=Du+rwx,Fu+rw --archive --delete \
      "${pkgs.ibisTestingData}/" \
      "$IBIS_TEST_DATA_DIRECTORY"

    export TEMPDIR
    TEMPDIR="$(python -c 'import tempfile; print(tempfile.gettempdir())')"
  '';

  nativeBuildInputs = devDeps ++ libraryDevDeps ++ [
    pythonEnv
    updateLockFiles
  ] ++ pkgs.preCommitShell.buildInputs ++ (with pkgs; [
    changelog
    mic
    poetry-cli
  ]);

  PYTHONPATH = builtins.toPath ./.;
  PGPASSWORD = "postgres";
  MYSQL_PWD = "ibis";
}
