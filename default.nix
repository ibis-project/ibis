{ python ? "3.10"
, doCheck ? true
}:
let
  pkgs = import ./nix;
  drv =
    { poetry2nix, python }:

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

      checkPhase = ''
        runHook preCheck
        pytest ibis/tests --numprocesses auto
        runHook postCheck
      '';

      inherit doCheck;

      pythonImportsCheck = [ "ibis" ];
    };
in
pkgs.callPackage drv {
  python = pkgs."python${builtins.replaceStrings [ "." ] [ "" ] python}";
}
