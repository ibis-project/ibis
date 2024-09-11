{
  description = "Expressive Python analytics at any scale.";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";

    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";

    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs = { self, flake-utils, gitignore, nixpkgs, poetry2nix, ... }: {
    overlays.default = nixpkgs.lib.composeManyExtensions [
      gitignore.overlay
      poetry2nix.overlays.default
      (import ./nix/overlay.nix)
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
      ];
      shellHook = ''
        rm -f "$PWD/ci/ibis-testing-data"
        ln -s "${pkgs.ibisTestingData}" "$PWD/ci/ibis-testing-data"

        # necessary for quarto and quartodoc
        export PYTHONPATH=''${PWD}''${PYTHONPATH:+:}''${PYTHONPATH}:''${PWD}/docs
      '';

      preCommitDeps = with pkgs; [
        actionlint
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
        name = "ibis-${env.python.version}";
        nativeBuildInputs = (with pkgs; [
          # python dev environment
          env
          # poetry executable
          poetry
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

        __darwinAllowLocalNetworking = true;
      };
    in
    rec {
      packages = {
        inherit (pkgs) ibis310 ibis311 ibis312;

        default = pkgs.ibis312;

        inherit (pkgs) update-lock-files check-release-notes-spelling;
      };

      devShells = rec {
        ibis310 = mkDevShell pkgs.ibisDevEnv310;
        ibis311 = mkDevShell pkgs.ibisDevEnv311;
        ibis312 = mkDevShell pkgs.ibisDevEnv312;

        # move back to 3.12 when dask-expr is supported or the dask backend is
        # removed
        default = ibis310;

        preCommit = pkgs.mkShell {
          name = "preCommit";
          nativeBuildInputs = [ pkgs.ibisSmallDevEnv ] ++ preCommitDeps;
        };

        links = pkgs.mkShell {
          name = "links";
          nativeBuildInputs = with pkgs; [ just lychee ];
        };

        release = pkgs.mkShell {
          name = "release";
          nativeBuildInputs = with pkgs; [ git poetry nodejs_20 unzip gnugrep ];
        };
      };
    }
  );
}
