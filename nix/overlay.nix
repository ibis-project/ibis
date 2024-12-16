{ uv2nix
, pyproject-nix
, pyproject-build-systems
}: pkgs: super:
let
  # Create package overlay from workspace.
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };

  envOverlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  # Create an overlay enabling editable mode for all local dependencies.
  # This is for usage with `nix develop`
  editableOverlay =
    workspace.mkEditablePyprojectOverlay {
      root = "$REPO_ROOT";
    };

  # Build fixups overlay
  pyprojectOverrides = import ./pyproject-overrides.nix { inherit pkgs; };

  # Adds tests to ibis-framework.passthru.tests
  testOverlay = import ./tests.nix {
    inherit pkgs;
    deps = defaultDeps;
  };

  # Default dependencies for env
  defaultDeps = {
    ibis-framework = [
      "duckdb"
      "datafusion"
      "sqlite"
      "polars"
      "decompiler"
      "visualization"
    ];
  };

  mkEnv' =
    {
      # Python dependency specification
      deps
    , # Installs ibis-framework as an editable package for use with `nix develop`.
      # This means that any changes done to your local files do not require a rebuild.
      editable
    ,
    }: python:
    let
      # Construct package set
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
          stdenv = pkgs.stdenv.override {
            targetPlatform = pkgs.stdenv.targetPlatform // {
              darwinSdkVersion = "12.0";
            };
          };
        }).overrideScope
          (lib.composeManyExtensions ([
            pyproject-build-systems.overlays.default
            envOverlay
            pyprojectOverrides
          ]
          ++ lib.optionals editable [ editableOverlay ]
          ++ lib.optionals (!editable) [ testOverlay ]));
    in
    # Build virtual environment
    (pythonSet.mkVirtualEnv "ibis-${python.pythonVersion}" deps).overrideAttrs (_old: {
      # Add passthru.tests from ibis-framework to venv passthru.
      # This is used to build tests by CI.
      passthru = {
        inherit (pythonSet.ibis-framework.passthru) tests;
      };
    });

  mkEnv = mkEnv' {
    deps = defaultDeps;
    editable = false;
  };

  mkDevEnv = mkEnv' {
    # Enable all dependencies for development shell
    deps = workspace.deps.all;
    editable = true;
  };

  inherit (pkgs) lib stdenv;
in
{
  ibisTestingData = pkgs.fetchFromGitHub {
    name = "ibis-testing-data";
    owner = "ibis-project";
    repo = "testing-data";
    rev = "b26bd40cf29004372319df620c4bbe41420bb6f8";
    sha256 = "sha256-1fenQNQB+Q0pbb0cbK2S/UIwZDE4PXXG15MH3aVbyLU=";
  };

  ibis310 = mkEnv pkgs.python310;
  ibis311 = mkEnv pkgs.python311;
  ibis312 = mkEnv pkgs.python312;
  ibis313 = mkEnv pkgs.python313;

  ibisDevEnv310 = mkDevEnv pkgs.python310;
  ibisDevEnv311 = mkDevEnv pkgs.python311;
  ibisDevEnv312 = mkDevEnv pkgs.python312;
  ibisDevEnv313 = mkDevEnv pkgs.python313;

  ibisSmallDevEnv = mkEnv'
    {
      deps = {
        ibis-framework = [ "dev" ];
      };
      editable = false;
    }
    pkgs.python313;

  duckdb = super.duckdb.overrideAttrs (
    _: lib.optionalAttrs (stdenv.isAarch64 && stdenv.isLinux) {
      doInstallCheck = false;
    }
  );

  quarto = pkgs.callPackage ./quarto { };

  changelog = pkgs.writeShellApplication {
    name = "changelog";
    runtimeInputs = [ pkgs.nodejs_20.pkgs.conventional-changelog-cli ];
    text = ''
      conventional-changelog --config ./.conventionalcommits.js "$@"
    '';
  };

  check-release-notes-spelling = pkgs.writeShellApplication {
    name = "check-release-notes-spelling";
    runtimeInputs = [ pkgs.changelog pkgs.coreutils pkgs.codespell ];
    text = ''
      tmp="$(mktemp)"
      changelog --release-count 1 --output-unreleased --outfile "$tmp"
      if ! codespell "$tmp"; then
        # cat -n to output line numbers
        cat -n "$tmp"
        exit 1
      fi
    '';
  };

  update-lock-files = pkgs.writeShellApplication {
    name = "update-lock-files";
    runtimeInputs = with pkgs; [ just uv ];
    text = "just lock";
  };

  gen-examples = pkgs.writeShellApplication {
    name = "gen-examples";
    runtimeInputs = [
      pkgs.ibisDevEnv313
      (pkgs.rWrapper.override {
        packages = with pkgs.rPackages; [
          Lahman
          janitor
          palmerpenguins
          stringr
          tidyverse
        ];
      })
      pkgs.google-cloud-sdk
    ];

    text = ''
      python "$PWD/ibis/examples/gen_registry.py" "''${@}"
    '';
  };
}
