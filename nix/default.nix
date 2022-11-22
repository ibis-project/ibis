let
  sources = import ./sources.nix;
in
{ ... }@args: import sources.nixpkgs ({
  overlays = [
    (import sources.rust-overlay)
    (pkgs: _: {
      inherit (import sources."gitignore.nix" {
        inherit (pkgs) lib;
      }) gitignoreSource;
    })
    (import "${sources.poetry2nix}/overlay.nix")
    (pkgs: super: {
      ibisTestingData = pkgs.fetchFromGitHub {
        owner = "ibis-project";
        repo = "testing-data";
        rev = "master";
        sha256 = "sha256-BZWi4kEumZemQeYoAtlUSw922p+R6opSWp/bmX0DjAo=";
      };

      rustNightly = pkgs.rust-bin.selectLatestNightlyWith (toolchain: toolchain.minimal);

      mkPoetryEnv = groups: python: pkgs.poetry2nix.mkPoetryEnv {
        inherit python groups;
        projectDir = ../.;
        editablePackageSources = {
          ibis = ../ibis;
        };
        overrides = pkgs.poetry2nix.overrides.withDefaults (
          import ../poetry-overrides.nix
        );
      };

      mkPoetryDocsEnv = pkgs.mkPoetryEnv [ "docs" ];
      mkPoetryDevEnv = pkgs.mkPoetryEnv [ "dev" "test" ];
      mkPoetryFullDevEnv = pkgs.mkPoetryEnv [ "dev" "docs" "test" ];

      prettierTOML = pkgs.writeShellScriptBin "prettier" ''
        ${pkgs.nodePackages.prettier}/bin/prettier \
        --plugin-search-dir "${pkgs.nodePackages.prettier-plugin-toml}/lib" \
        "$@"
      '';

      ibisDevEnv38 = pkgs.mkPoetryDevEnv pkgs.python38;
      ibisDevEnv39 = pkgs.mkPoetryDevEnv pkgs.python39;
      ibisDevEnv310 = pkgs.mkPoetryDevEnv pkgs.python310;

      ibisDevEnv = pkgs.ibisDevEnv310;

      ibisDocsEnv38 = pkgs.mkPoetryDocsEnv pkgs.python38;
      ibisDocsEnv39 = pkgs.mkPoetryDocsEnv pkgs.python39;
      ibisDocsEnv310 = pkgs.mkPoetryDocsEnv pkgs.python310;

      ibisDocsEnv = pkgs.ibisDocsEnv310;

      ibisFullDevEnv38 = pkgs.mkPoetryFullDevEnv pkgs.python38;
      ibisFullDevEnv39 = pkgs.mkPoetryFullDevEnv pkgs.python39;
      ibisFullDevEnv310 = pkgs.mkPoetryFullDevEnv pkgs.python310;

      ibisFullDevEnv = pkgs.ibisFullDevEnv310;

      changelog = pkgs.writeShellApplication {
        name = "changelog";
        runtimeInputs = [ pkgs.nodePackages.conventional-changelog-cli ];
        text = ''
          conventional-changelog --config ./.conventionalcommits.js
        '';
      };

      preCommitShell = pkgs.mkShell {
        name = "preCommitShell";
        buildInputs = with pkgs; [
          actionlint
          git
          just
          nix-linter
          nixpkgs-fmt
          pre-commit
          prettierTOML
          shellcheck
          shfmt
        ];
      };

      arrow-cpp = (super.arrow-cpp.override {
        enableS3 = !super.stdenv.isDarwin;
      }).overrideAttrs
        (attrs: {
          # tests hang for some reason on macos with python 3.9 and 3.10
          doInstallCheck = !super.stdenv.isDarwin;
          buildInputs = attrs.buildInputs or [ ]
            ++ pkgs.lib.optionals super.stdenv.isDarwin [
            pkgs.libxcrypt
            pkgs.sqlite
          ];
        });
    })
  ];
} // args)
