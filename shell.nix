{ python ? "3.10" }:
let
  pkgs = import ./nix;

  devDeps = with pkgs; [
    cacert
    cachix
    commitlint
    curl
    duckdb
    git
    glow
    jq
    just
    lychee
    mariadb
    niv
    nix-linter
    nixpkgs-fmt
    poetry
    prettierTOML
    shellcheck
    shfmt
    unzip
    yj
  ];

  impalaUdfDeps = with pkgs; [
    clang_12
    cmake
    ninja
  ];

  backendTestDeps = [ pkgs.docker-compose_2 ];
  vizDeps = [ pkgs.graphviz-nox ];
  pysparkDeps = [ pkgs.openjdk11_headless ];
  docDeps = [ pkgs.pandoc ];

  # postgresql is the client, not the server
  postgresDeps = [ pkgs.postgresql ];
  geospatialDeps = with pkgs; [ gdal_2 proj ];

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
  changelog = pkgs.writeShellApplication {
    name = "changelog";
    runtimeInputs = [ pkgs.nodePackages.conventional-changelog-cli ];
    text = ''
      conventional-changelog --preset conventionalcommits
    '';
  };
  mic = pkgs.writeShellApplication {
    name = "mic";
    runtimeInputs = [ pythonEnv pkgs.coreutils ];
    text = ''
      # The immediate reason this is necessary is to allow the subprocess
      # invocations of `mkdocs` by `mike` to see Python dependencies.
      #
      # This shouldn't be necessary, but I think the nix wrappers may be
      # indavertently preventing this.
      export PYTHONPATH TEMPDIR
      PYTHONPATH="$(python -c 'import os, sys; print(os.pathsep.join(sys.path))')"
      TEMPDIR="$(python -c 'import tempfile; print(tempfile.gettempdir())')"

      mike "$@"
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
  '';

  buildInputs = devDeps ++ libraryDevDeps ++ [
    changelog
    mic
    pythonEnv
    (pythonEnv.python.pkgs.toPythonApplication pkgs.pre-commit)
  ];

  PYTHONPATH = builtins.toPath ./.;
}
