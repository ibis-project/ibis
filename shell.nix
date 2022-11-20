{ python ? "3.10" }:
let
  pkgs = import ./nix { };

  pythonShortVersion = builtins.replaceStrings [ "." ] [ "" ] python;

  pythonEnv = pkgs."ibisFullDevEnv${pythonShortVersion}";

  devDeps = with pkgs; [
    # terminal markdown rendering
    glow
    # used in the justfile
    jq
    yj
    # linting
    commitlint
    lychee
    # external nix dependencies
    niv
  ];

  impalaUdfDeps = with pkgs; [ clang_12 cmake ninja ];
  snowflakeDeps = [ pkgs.openssl ];
  backendTestDeps = [ pkgs.docker-compose ];
  vizDeps = [ pkgs.graphviz-nox ];
  duckdbDeps = [ pkgs.duckdb ];
  mysqlDeps = [ pkgs.mariadb-client ];
  pysparkDeps = [ pkgs.openjdk11_headless ];

  postgresDeps = [ pkgs.postgresql ];
  geospatialDeps = with pkgs; [ gdal proj ];
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
  ] ++ pkgs.preCommitShell.buildInputs ++ (with pkgs; [
    changelog
    mic
  ]);

  PGPASSWORD = "postgres";
  MYSQL_PWD = "ibis";
  MSSQL_SA_PASSWORD = "1bis_Testing!";
}
