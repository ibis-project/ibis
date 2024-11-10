{ uv2nix, pyproject-nix }: pkgs: super:
let
  # Create package overlay from workspace.
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };

  envOverlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  editableOverlay =
    # Create an overlay enabling editable mode for all local dependencies.
    # This is for usage with nix-nix develop
    workspace.mkEditablePyprojectOverlay {
      root = "$REPO_ROOT";
    };

  pyprojectOverrides = import ./pyproject-overrides.nix { inherit pkgs; };

  mkEnv = python: { deps ? workspace.deps.all, editable ? true }:
    # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
    #
    # This means that any changes done to your local files do not require a rebuild.
    let
      # Construct package set
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (lib.composeManyExtensions ([
            envOverlay
            pyprojectOverrides
          ] ++ lib.optional editable editableOverlay));
    in
    # Build virtual environment
    pythonSet.mkVirtualEnv "ibis-${python.pythonVersion}" deps;

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

  ibis310 = pkgs.callPackage ./ibis.nix {
    python = pkgs.python310;
    inherit mkEnv;
  };

  ibis311 = pkgs.callPackage ./ibis.nix {
    python = pkgs.python311;
    inherit mkEnv;
  };

  ibis312 = pkgs.callPackage ./ibis.nix {
    python = pkgs.python312;
    inherit mkEnv;
  };

  ibisDevEnv310 = mkEnv pkgs.python310 { };
  ibisDevEnv311 = mkEnv pkgs.python311 { };
  ibisDevEnv312 = mkEnv pkgs.python312 { };

  ibisSmallDevEnv = mkEnv pkgs.python312 {
    deps = {
      ibis-framework = [ "dev" ];
    };
  };

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
      pkgs.ibisDevEnv312
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
