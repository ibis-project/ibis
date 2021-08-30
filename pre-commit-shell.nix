{ python ? "3.10" }:
let
  pkgs = import ./nix;
  inherit (pkgs) lib;

  shellHook = ''
    ${(import ./pre-commit.nix).pre-commit-check.shellHook}
  '';
  pythonShortVersion = builtins.replaceStrings [ "." ] [ "" ] python;
in
pkgs.mkShell {
  name = "ibis-pre-commit-${pythonShortVersion}";

  inherit shellHook;

  buildInputs = (with pkgs; [
    git
    nix-linter
    nixpkgs-fmt
    shellcheck
    shfmt
  ]) ++ [
    pkgs."ibisDevEnv${pythonShortVersion}"
  ];
}
