{ lib, pkgs, stdenv, ... }:
let
  numpyVersion = self: self.numpy.version;
in
self: super:
{
  # see https://github.com/numpy/numpy/issues/19624 for details
  numpy = super.numpy.overridePythonAttrs (attrs: {
    patches = (attrs.patches or [ ])
      ++ lib.optional
      # this patch only applies to macos and only with numpy versions >=1.21,<1.21.2
      (stdenv.isDarwin && (lib.versionAtLeast (numpyVersion self) "1.21.0" && lib.versionOlder (numpyVersion self) "1.21.2"))
      (pkgs.fetchpatch {
        url = "https://github.com/numpy/numpy/commit/8045183084042fbafc995dd26eb4d9ca45eb630e.patch";
        sha256 = "14g69vq7llkh6smpfrb50iqga7fd360dkcc0rlwb5k2cz8bsii5b";
      });
  });

  # TODO: remove this entire override when upstream nixpkgs datafusion PR
  # https://github.com/NixOS/nixpkgs/pull/152763 is merged
  datafusion =
    let
      cargoLock = pkgs.fetchurl {
        url = "https://raw.githubusercontent.com/apache/arrow-datafusion/6.0.0/python/Cargo.lock";
        sha256 = "sha256-xiv3drEU5jOGsEIh0U01ZQ1NBKobxO2ctp4mxy9iigw=";
      };

      postUnpack = ''
        # TODO: apache/arrow-datafusion should ship the Cargo.lock file in the Python package
        cp "${cargoLock}" $sourceRoot/Cargo.lock
        chmod u+w $sourceRoot/Cargo.lock
      '';
    in
    super.datafusion.overridePythonAttrs (attrs: {
      nativeBuildInputs = (attrs.nativeBuildInputs or [ ])
        ++ (with pkgs.rustPlatform; [ cargoSetupHook maturinBuildHook ]);

      buildInputs = (attrs.buildInputs or [ ])
        ++ lib.optionals stdenv.isDarwin [ pkgs.libiconv ];

      inherit postUnpack;

      patches = [ ./patches/Cargo.lock.patch ];

      cargoDeps = pkgs.rustPlatform.fetchCargoTarball {
        inherit (attrs) src;
        inherit postUnpack;
        sourceRoot = "${attrs.pname}-${attrs.version}";
        patches = [ ./patches/Cargo.lock.patch ];
        sha256 = "sha256-JGyDxpfBXzduJaMF1sbmRm7KJajHYdVSj+WbiSETiY0=";
      };
    });

  mkdocs-literate-nav = super.mkdocs-literate-nav.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [ self.poetry ];
  });

  pybind11 = super.pybind11.overridePythonAttrs (_: {
    postBuild = ''
      make -j $NIX_BUILD_CORES -l $NIX_BUILD_CORES
    '';
  });

  duckdb = super.duckdb.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
      self.pybind11
    ];
  });

  duckdb-engine = super.duckdb-engine.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
      self.poetry-core
    ];
  });

  lz4 = super.lz4.overridePythonAttrs (
    attrs: lib.optionalAttrs (lib.versionOlder super.lz4.version "4.0.0") {
      nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
        self.pkgconfig
      ];
    }
  );

  mkdocs-gen-files = super.mkdocs-gen-files.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
      self.poetry
    ];
  });
}
