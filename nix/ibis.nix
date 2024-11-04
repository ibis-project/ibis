{ pkgs, python, mkEnv, stdenv, ... }:
let
  pythonEnv = mkEnv python {
    editable = false;
    deps = {
      ibis-framework = [
        # Groups
        "tests"
        # Extras
        "duckdb"
        "datafusion"
        "sqlite"
        "polars"
        "decompiler"
        "visualization"
      ];
    };
  };

in
stdenv.mkDerivation {
  name = "ibis-framework-test";
  nativeCheckInputs = [ pythonEnv pkgs.graphviz-nox ];
  src = ../.;
  doCheck = true;
  preCheck = ''
    ln -s ${pkgs.ibisTestingData} $PWD/ci/ibis-testing-data
  '';
  checkPhase = ''
    runHook preCheck
    pytest -m datafusion
    pytest -m 'core or duckdb or sqlite or polars' --numprocesses $NIX_BUILD_CORES --dist loadgroup
    runHook postCheck
  '';

  # Don't run the fixup phase(s), to avoid permissions errors
  dontFixup = true;

  # ibis-framework was already built as a part of the env, this is just running
  # tests. Symlink the built test env for convenience.
  #
  # Note: Testing could technically be done as a part of the virtualenv
  # constructor derivation.
  installPhase = ''
    runHook preInstall
    ln -s ${pythonEnv} $out
    runHook postInstall
  '';
}
