{ python ? "3.10" }:
let
  pkgs = import ./nix;
  inherit (pkgs) lib;

  devDeps = with pkgs; [
    cacert
    cachix
    commitlint
    gh
    git
    niv
    nix-linter
    nixpkgs-fmt
    poetry
    shellcheck
    shfmt
  ];

  impalaUdfDeps = with pkgs; [
    boost
    clang_12
    cmake
  ];

  vizDeps = [ pkgs.graphviz ];
  pysparkDeps = [ pkgs.openjdk11 ];
  docDeps = [ pkgs.pandoc ];

  # postgresql is the client, not the server
  postgresDeps = [ pkgs.postgresql ];

  sqliteDeps = [ pkgs.sqlite-interactive ];

  libraryDevDeps = impalaUdfDeps
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
    ${(import ./pre-commit.nix).pre-commit-check.shellHook}
    data_dir="ci/ibis-testing-data"
    mkdir -p "$data_dir"
    chmod u+rwx "$data_dir"
    cp -rf ${pkgs.ibisTestingData}/* "$data_dir"
    chmod --recursive u+rw "$data_dir"
  '';

  buildInputs = devDeps ++ libraryDevDeps ++ [
    pkgs."ibisDevEnv${pythonShortVersion}"
  ];
}
