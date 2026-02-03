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
    {
      self,
      flake-utils,
      gitignore,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    {
      overlays.default = nixpkgs.lib.composeManyExtensions [
        gitignore.overlay
        (import ./nix/overlay.nix { inherit uv2nix pyproject-nix pyproject-build-systems; })
      ];
    }
    // flake-utils.lib.eachDefaultSystem (
      localSystem:
      let
        pkgs = import nixpkgs {
          inherit localSystem;
          overlays = [ self.overlays.default ];
        };

        backendDevDeps = [
          # impala UDFs
          pkgs.clang_18
          pkgs.cmake
          pkgs.ninja
          # snowflake
          pkgs.openssl
          # backend test suite
          pkgs.docker-compose
          # visualization
          pkgs.graphviz-nox
          # duckdb
          pkgs.duckdb
          # mysql
          pkgs.mariadb.client
          # pyodbc setup debugging
          # in particular: odbcinst -j
          pkgs.unixODBC
          # pyspark
          pkgs.openjdk17_headless
          # postgres client
          pkgs.libpq.pg_config
          pkgs.postgresql
          # sqlite with readline
          pkgs.sqlite-interactive
          # mysqlclient build
          pkgs.libmysqlclient
          pkgs.pkg-config
          # new hotness for build orchestration (?)
          pkgs.docker-buildx
          # toml setting
          pkgs.toml-cli
          # sponge
          pkgs.moreutils
        ];
        shellHook = ''
          rm -f "$PWD/ci/ibis-testing-data"
          ln -s "${pkgs.ibisTestingData}" "$PWD/ci/ibis-testing-data"

          # Undo dependency propagation by nixpkgs.
          unset PYTHONPATH

          # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
          export REPO_ROOT=$(git rev-parse --show-toplevel)

          # pure python implementation of psycopg needs to be able to find
          # libpq
          #
          # AFAIK there isn't another way to force the lookup to occur where it
          # needs to in the nix store without setting LD_LIBRARY_PATH
          export LD_LIBRARY_PATH="$(pg_config --libdir)"
        '';

        preCommitDeps =
          pkgs.lib.optionals (!pkgs.stdenv.isDarwin) [
            pkgs.actionlint
          ]
          ++ [
            pkgs.codespell
            pkgs.deadnix
            pkgs.git
            pkgs.just
            pkgs.nixfmt
            pkgs.nodejs.pkgs.prettier
            pkgs.shellcheck
            pkgs.shfmt
            pkgs.statix
            pkgs.taplo
          ];

        mkDevShell =
          env:
          pkgs.mkShell {
            inherit (env) name;
            packages = [
              # python dev environment
              env
            ]
            ++ [
              # uv executable
              pkgs.uv
              # rendering release notes
              pkgs.changelog
              pkgs.glow
              # used in the justfile
              pkgs.jq
              pkgs.yj
              # commit linting
              pkgs.commitlint
              # link checking
              pkgs.lychee
              # release automation
              pkgs.nodejs
              # used in notebooks to download data
              pkgs.curl
              # docs
              pkgs.quarto
            ]
            ++ preCommitDeps
            ++ backendDevDeps;

            inherit shellHook;

            PYSPARK_PYTHON = "${env}/bin/python";

            AWS_PROFILE = "ibis-testing";

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
          default = packages.ibis314;

          inherit (pkgs)
            ibis310
            ibis311
            ibis312
            ibis313
            ibis314
            check-release-notes-spelling
            get-latest-quarto-hash
            ;
        };

        checks = {
          ibis310-pytest = pkgs.ibis310.passthru.tests.pytest;
          ibis311-pytest = pkgs.ibis311.passthru.tests.pytest;
          ibis312-pytest = pkgs.ibis312.passthru.tests.pytest;
          ibis313-pytest = pkgs.ibis313.passthru.tests.pytest;
          ibis314-pytest = pkgs.ibis314.passthru.tests.pytest;
        };

        devShells = rec {
          ibis310 = mkDevShell pkgs.ibisDevEnv310;
          ibis311 = mkDevShell pkgs.ibisDevEnv311;
          ibis312 = mkDevShell pkgs.ibisDevEnv312;
          ibis313 = mkDevShell pkgs.ibisDevEnv313;
          ibis314 = mkDevShell pkgs.ibisDevEnv314;

          default = ibis314;

          preCommit = pkgs.mkShell {
            name = "preCommit";
            packages = preCommitDeps ++ [ pkgs.ibisSmallDevEnv ];
          };

          links = pkgs.mkShell {
            name = "links";
            packages = [
              pkgs.just
              pkgs.lychee
            ];
          };

          release = pkgs.mkShell {
            name = "release";
            packages = [
              pkgs.git
              pkgs.uv
              pkgs.nodejs
              pkgs.unzip
              pkgs.gnugrep
              (pkgs.python3.withPackages (p: [ p.packaging ]))
            ];
          };
        };
      }
    );
}
