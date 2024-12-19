{
  description = "Expressive Python analytics at any scale.";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";

    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";

    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:adisbladis/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs =
    { self
    , flake-utils
    , gitignore
    , nixpkgs
    , pyproject-nix
    , uv2nix
    , pyproject-build-systems
    , ...
    }: {
      overlays.default = nixpkgs.lib.composeManyExtensions [
        gitignore.overlay
        (import ./nix/overlay.nix { inherit uv2nix pyproject-nix pyproject-build-systems; })
      ];
    } // flake-utils.lib.eachDefaultSystem (
      localSystem:
      let
        pkgs = import nixpkgs {
          inherit localSystem;
          overlays = [ self.overlays.default ];
        };

        backendDevDeps = with pkgs; [
          # impala UDFs
          clang_15
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
          # pyodbc setup debugging
          # in particular: odbcinst -j
          unixODBC
          # pyspark
          openjdk17_headless
          # postgres client
          postgresql
          # sqlite with readline
          sqlite-interactive
          # mysqlclient build
          libmysqlclient
          pkg-config
        ];
        shellHook = ''
          rm -f "$PWD/ci/ibis-testing-data"
          ln -s "${pkgs.ibisTestingData}" "$PWD/ci/ibis-testing-data"

          # Undo dependency propagation by nixpkgs.
          unset PYTHONPATH

          # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
          export REPO_ROOT=$(git rev-parse --show-toplevel)
        '';

        preCommitDeps = with pkgs; [
          actionlint
          codespell
          deadnix
          git
          just
          nixpkgs-fmt
          nodejs_20.pkgs.prettier
          shellcheck
          shfmt
          statix
          taplo-cli
        ];

        mkDevShell = env: pkgs.mkShell {
          inherit (env) name;
          packages = [
            # python dev environment
            env
          ] ++ (with pkgs; [
            # uv executable
            uv
            # rendering release notes
            changelog
            glow
            # used in the justfile
            jq
            yj
            # commit linting
            commitlint
            # link checking
            lychee
            # release automation
            nodejs_20
            # used in notebooks to download data
            curl
            # docs
            quarto
          ])
          ++ preCommitDeps
          ++ backendDevDeps;

          inherit shellHook;

          PYSPARK_PYTHON = "${env}/bin/python";

          # needed for mssql+pyodbc
          ODBCSYSINI = pkgs.writeTextDir "odbcinst.ini" ''
            [FreeTDS]
            Driver = ${pkgs.lib.makeLibraryPath [ pkgs.freetds ]}/libtdsodbc.so
          '';

          GDAL_DATA = "${pkgs.gdal}/share/gdal";
          PROJ_DATA = "${pkgs.proj}/share/proj";

          __darwinAllowLocalNetworking = true;
        };
      in
      rec {
        packages = {
          default = packages.ibis313;

          inherit (pkgs) ibis310 ibis311 ibis312 ibis313
            update-lock-files check-release-notes-spelling;
        };

        checks = {
          ibis310-pytest = pkgs.ibis310.passthru.tests.pytest;
          ibis311-pytest = pkgs.ibis311.passthru.tests.pytest;
          ibis312-pytest = pkgs.ibis312.passthru.tests.pytest;
          ibis313-pytest = pkgs.ibis313.passthru.tests.pytest;
        };

        devShells = rec {
          ibis310 = mkDevShell pkgs.ibisDevEnv310;
          ibis311 = mkDevShell pkgs.ibisDevEnv311;
          ibis312 = mkDevShell pkgs.ibisDevEnv312;
          ibis313 = mkDevShell pkgs.ibisDevEnv313;

          default = ibis312;

          preCommit = pkgs.mkShell {
            name = "preCommit";
            packages = preCommitDeps ++ [ pkgs.ibisSmallDevEnv ];
          };

          links = pkgs.mkShell {
            name = "links";
            packages = with pkgs; [ just lychee ];
          };

          release = pkgs.mkShell {
            name = "release";
            packages = with pkgs; [
              git
              uv
              nodejs_20
              unzip
              gnugrep
              (python3.withPackages (p: [ p.packaging ]))
            ];
          };
        };
      }
    );
}
