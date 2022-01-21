{ python ? "3.9" }:
let
  pkgs = import ./nix;

  devDeps = with pkgs; [
    cacert
    cachix
    commitlint
    git
    niv
    nix-linter
    nixpkgs-fmt
    poetry
    pre-commit
    prettierTOML
    shellcheck
    shfmt
  ];

  impalaUdfDeps = with pkgs; [
    boost
    clang_12
    cmake
  ];

  backendTestDeps = [ pkgs.docker-compose ];
  vizDeps = [ pkgs.graphviz-nox ];
  pysparkDeps = [ pkgs.openjdk11 ];
  docDeps = [ pkgs.pandoc ];

  # postgresql is the client, not the server
  postgresDeps = [ pkgs.postgresql ];

  sqliteDeps = [ pkgs.sqlite-interactive ];

  libraryDevDeps = impalaUdfDeps
    ++ backendTestDeps
    ++ vizDeps
    ++ pysparkDeps
    ++ docDeps
    ++ postgresDeps
    ++ sqliteDeps;

  pythonShortVersion = builtins.replaceStrings [ "." ] [ "" ] python;
in
pkgs.mkShell {
  name = "ibis${pythonShortVersion}";

  shellHook = ''
    data_dir="ci/ibis-testing-data"
    mkdir -p "$data_dir"
    chmod u+rwx "$data_dir"
    cp -rf ${pkgs.ibisTestingData}/* "$data_dir"
    chmod --recursive u+rw "$data_dir"
  '';

  buildInputs = devDeps ++ libraryDevDeps ++ [
    pkgs."ibisDevEnv${pythonShortVersion}"
  ];

  PYTHONPATH = builtins.toPath ./.;
}
