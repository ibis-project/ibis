let
  sources = import ./sources.nix;
in
import sources.nixpkgs {
  overlays = [
    (pkgs: _: {
      inherit (import sources."gitignore.nix" {
        inherit (pkgs) lib;
      }) gitignoreSource;
    })
    (pkgs: super: {
      poetry2nix = import sources.poetry2nix {
        inherit pkgs;
        inherit (pkgs) poetry;
      };

      ibisTestingData = pkgs.fetchFromGitHub {
        owner = "ibis-project";
        repo = "testing-data";
        rev = "master";
        sha256 = "1lm66g5kvisxsjf1jwayiyxl2d3dhlmxj13ijrya3pfg07mq9r66";
      };

      mkPoetryEnv = python: pkgs.poetry2nix.mkPoetryEnv {
        inherit python;
        projectDir = ../.;
        editablePackageSources = {
          ibis = ../ibis;
        };
        overrides = pkgs.poetry2nix.overrides.withDefaults (
          import ../poetry-overrides.nix {
            inherit pkgs;
            inherit (pkgs) lib stdenv;
          }
        );
      };

      prettierTOML = pkgs.writeShellScriptBin "prettier" ''
        ${pkgs.nodePackages.prettier}/bin/prettier \
        --plugin-search-dir "${pkgs.nodePackages.prettier-plugin-toml}/lib" \
        "$@"
      '';

      ibisDevEnv38 = pkgs.mkPoetryEnv pkgs.python38;
      ibisDevEnv39 = pkgs.mkPoetryEnv pkgs.python39;
      ibisDevEnv310 = pkgs.mkPoetryEnv pkgs.python310;
      ibisDevEnv = pkgs.ibisDevEnv310;
    } // super.lib.optionalAttrs super.stdenv.isDarwin {
      arrow-cpp = super.arrow-cpp.override {
        enableS3 = false;
      };
    })
  ];
}
