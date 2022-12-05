{
  description = "Expressive Python analytics at any scale.";

  inputs = {
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };

    flake-utils.url = "github:numtide/flake-utils";

    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";

    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs =
    { self
    , flake-utils
    , gitignore
    , nixpkgs
    , poetry2nix
    , rust-overlay
    , ...
    }:
    let
      backends = [
        "dask"
        "datafusion"
        "duckdb"
        "pandas"
        "polars"
        "sqlite"
      ];
      drv =
        { poetry2nix
        , python
        , lib
        , gitignoreSource
        , gdal
        , graphviz-nox
        , proj
        , sqlite
        , rsync
        , ibisTestingData
        , ...
        }: poetry2nix.mkPoetryApplication rec {
          inherit python;

          groups = [ ];
          checkGroups = [ "test" ];
          projectDir = gitignoreSource ./.;
          src = gitignoreSource ./.;

          overrides = poetry2nix.overrides.withDefaults (
            import ./poetry-overrides.nix
          );

          buildInputs = [ gdal graphviz-nox proj sqlite ];
          checkInputs = buildInputs;

          preCheck = ''
            set -euo pipefail

            export IBIS_TEST_DATA_DIRECTORY="$PWD/ci/ibis-testing-data"

            ${rsync}/bin/rsync \
              --chmod=Du+rwx,Fu+rw --archive --delete \
              "${ibisTestingData}/" \
              "$IBIS_TEST_DATA_DIRECTORY"
          '';

          checkPhase = ''
            set -euo pipefail

            runHook preCheck

            pytest --numprocesses auto --dist loadgroup -m '${lib.concatStringsSep " or " backends} or core'

            runHook postCheck
          '';

          doCheck = true;

          pythonImportsCheck = [ "ibis" ] ++ (map (backend: "ibis.backends.${backend}") backends);
        };
    in
    {
      overlay = nixpkgs.lib.composeManyExtensions [
        gitignore.overlay
        poetry2nix.overlay
        rust-overlay.overlays.default
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
            projectDir = pkgs.gitignoreSource ./.;
            editablePackageSources = { ibis = pkgs.gitignoreSource ./ibis; };
            overrides = pkgs.poetry2nix.overrides.withDefaults (import ./poetry-overrides.nix);
          };

          mkPoetryDocsEnv = pkgs.mkPoetryEnv [ "docs" ];
          mkPoetryDevEnv = pkgs.mkPoetryEnv [ "dev" "test" ];
          mkPoetryFullDevEnv = pkgs.mkPoetryEnv [ "dev" "docs" "test" ];

          prettierTOML = pkgs.writeShellScriptBin "prettier" ''
            ${pkgs.nodePackages.prettier}/bin/prettier \
            --plugin-search-dir "${pkgs.nodePackages.prettier-plugin-toml}/lib" "$@"
          '';

          ibis38 = pkgs.callPackage drv { python = pkgs.python38; };
          ibis39 = pkgs.callPackage drv { python = pkgs.python39; };
          ibis310 = pkgs.callPackage drv { python = pkgs.python310; };

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
            text = "conventional-changelog --config ./.conventionalcommits.js";
          };

          arrow-cpp = (super.arrow-cpp.override {
            enableS3 = !super.stdenv.isDarwin;
          }).overrideAttrs
            (attrs: {
              # tests hang for some reason on macos with python 3.9 and 3.10
              doInstallCheck = !super.stdenv.isDarwin;
              buildInputs = attrs.buildInputs or [ ] ++ pkgs.lib.optionals super.stdenv.isDarwin [
                pkgs.libxcrypt
                pkgs.sqlite
              ];
            });
        })
      ];
    } // flake-utils.lib.eachDefaultSystem (
      localSystem:
      let
        pkgs = import nixpkgs {
          inherit localSystem;
          overlays = [ self.overlay ];
        };
        inherit (pkgs) lib;

        backendDevDeps = with pkgs; [
          # impala UDFs
          clang_12
          cmake
          ninja
          # snowflake
          openssl
          # backend test suite
          docker-compose
          # visualization
          graphviz-nox
          # duckdb
          duckdb
          # mysql
          mariadb-client
          # pyspark
          openjdk11_headless
          # postgres
          postgresql
          # geospatial
          gdal
          proj
          # sqlite
          sqlite-interactive
        ];
        mkDevShell = env: pkgs.mkShell {
          nativeBuildInputs = (with pkgs; [
            # python dev environment
            env
            # rendering release notes
            changelog
            glow
            # used in the justfile
            jq
            yj
            # linting
            commitlint
            lychee
            # release automation
            nodejs
            # poetry executable
            env.pkgs.poetry
            env.pkgs.poetry-core
            # pre-commit deps
            actionlint
            git
            just
            nix-linter
            nixpkgs-fmt
            pre-commit
            prettierTOML
            shellcheck
            shfmt
          ])
          # backend development dependencies
          ++ backendDevDeps;

          shellHook = ''
            export IBIS_TEST_DATA_DIRECTORY="$PWD/ci/ibis-testing-data"

            ${pkgs.rsync}/bin/rsync \
              --chmod=Du+rwx,Fu+rw --archive --delete \
              "${pkgs.ibisTestingData}/" \
              "$IBIS_TEST_DATA_DIRECTORY"

            export TEMPDIR
            TEMPDIR="$(python -c 'import tempfile; print(tempfile.gettempdir())')"
          '';

          PGPASSWORD = "postgres";
          MYSQL_PWD = "ibis";
          MSSQL_SA_PASSWORD = "1bis_Testing!";
          # necessary for mkdocs
          PYTHONPATH = builtins.toPath ./.;
        };
      in
      rec {
        packages = rec {
          ibis38 = pkgs.ibis38;
          ibis39 = pkgs.ibis39;
          ibis310 = pkgs.ibis310;
          default = ibis310;
        };

        devShells = rec {
          ibis38 = mkDevShell pkgs.ibisFullDevEnv38;
          ibis39 = mkDevShell pkgs.ibisFullDevEnv39;
          ibis310 = mkDevShell pkgs.ibisFullDevEnv310;
          default = ibis310;
        };
      }
    );
}
