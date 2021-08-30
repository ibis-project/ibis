let
  inherit (import ./nix) lib;
  sources = import ./nix/sources.nix;
  pre-commit-hooks = import sources.pre-commit-hooks;
in
{
  pre-commit-check = pre-commit-hooks.run {
    src = ./.;
    hooks = {
      black = {
        enable = true;
        entry = lib.mkForce "black --check";
        types = [ "python" ];
      };

      isort = {
        enable = true;
        language = "python";
        entry = lib.mkForce "isort --check";
        types_or = [ "cython" "pyi" "python" ];
      };

      flake8 = {
        enable = true;
        language = "python";
        entry = "flake8";
        types = [ "python" ];
      };

      nix-linter = {
        enable = true;
        entry = lib.mkForce "nix-linter";
        excludes = [
          "nix/sources.nix"
        ];
      };

      nixpkgs-fmt = {
        enable = true;
        entry = lib.mkForce "nixpkgs-fmt --check";
        excludes = [
          "nix/sources.nix"
        ];
      };

      shellcheck = {
        enable = true;
        entry = lib.mkForce "shellcheck";
        files = "\\.sh$";
        types_or = lib.mkForce [ ];
      };

      shfmt = {
        enable = true;
        entry = lib.mkForce "shfmt -i 2 -sr -d -s -l";
        files = "\\.sh$";
      };

      pyupgrade = {
        enable = true;
        entry = "pyupgrade --py37-plus";
        types = [ "python" ];
        excludes = [ "setup.py" ];
      };
    };
  };
}
