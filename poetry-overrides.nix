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

  datafusion = super.datafusion.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ])
      ++ (with pkgs.rustPlatform; [ cargoSetupHook maturinBuildHook ]);

    buildInputs = (attrs.buildInputs or [ ])
      ++ lib.optionals stdenv.isDarwin [ pkgs.libiconv ];

    cargoDeps = pkgs.rustPlatform.fetchCargoTarball {
      inherit (attrs) src;
      sourceRoot = "${attrs.pname}-${attrs.version}";
      sha256 = "sha256-ITABYhcTZZd21Eg7YI/vbO2WiBnanidZ/08K3sD8GnQ=";
    };
  });

  pybind11 = super.pybind11.overridePythonAttrs (_: {
    postBuild = ''
      make -j $NIX_BUILD_CORES -l $NIX_BUILD_CORES
    '';
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

  mkdocs-autorefs = super.mkdocs-autorefs.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
      self.pdm-pep517
    ];
  });

  mkdocs-jupyter = super.mkdocs-jupyter.overridePythonAttrs (attrs: {
    propagatedBuildInputs = (attrs.propagatedBuildInputs or [ ]) ++ [
      self.ipython_genutils
    ];
  });
}
