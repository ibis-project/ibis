{ pkgs, deps }:
let
  inherit (pkgs) stdenv;
in
final: prev: {
  ibis-framework = prev.ibis-framework.overrideAttrs (old: {
    passthru = old.passthru // {
      tests = old.passthru.tests or { } // {
        pytest =
          let
            pythonEnv = final.mkVirtualEnv "ibis-framework-test-env" (deps // {
              # Use default dependencies from overlay.nix + enabled tests group.
              ibis-framework = deps.ibis-framework or [ ] ++ [ "tests" ];
            });
          in
          stdenv.mkDerivation {
            name = "ibis-framework-test";
            nativeCheckInputs = [ pythonEnv pkgs.graphviz-nox ];
            src = ../.;
            doCheck = true;
            preCheck = ''
              set -euo pipefail

              ln -s ${pkgs.ibisTestingData} $PWD/ci/ibis-testing-data

              HOME="$(mktemp -d)"
              export HOME
            '';
            checkPhase = ''
              runHook preCheck
              pytest -m datafusion
              pytest -m 'core or duckdb or sqlite or polars' --numprocesses $NIX_BUILD_CORES --dist loadgroup
              runHook postCheck
            '';

            installPhase = "mkdir $out";
          };
      };
    };
  });
}
