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

  krb5 = super.krb5.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ])
      ++ [ pkgs.krb5Full ]
      # AppKit includes Heimdal, which is required for krb5 support on MacOS
      ++ lib.optional stdenv.isDarwin [ pkgs.darwin.apple_sdk.frameworks.AppKit ];

    # -fno-strict-overflow is not a supported argument in clang on darwin
    hardeningDisable = lib.optionals stdenv.isDarwin [ "strictoverflow" ];
  } // lib.optionalAttrs stdenv.isDarwin {
    KRB5_LINKER_ARGS = " ";
  });

  gssapi = super.gssapi.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ])
      ++ [ self.cython pkgs.krb5Full ];

    buildInputs = (attrs.buildInputs or [ ])
      ++ lib.optionals stdenv.isDarwin [ pkgs.darwin.apple_sdk.frameworks.GSS ];

    postPatch = ''
      substituteInPlace setup.py \
        --replace 'get_output(f"{kc} gssapi --prefix")' '"${lib.getDev pkgs.krb5Full}"'
    '';
  });

  tables = super.tables.overridePythonAttrs (attrs: {
    format = "setuptools";

    buildInputs = (attrs.buildInputs or [ ]) ++ (with pkgs; [ bzip2 c-blosc hdf5 lzo ]);
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [ self.cython ];

    # Regenerate C code with Cython
    preBuild = ''
      make distclean
    '';

    # When doing `make distclean`, ignore docs
    postPatch = ''
      substituteInPlace Makefile --replace "src doc" "src"
      # Force test suite to error when unittest runner fails
      substituteInPlace tables/tests/test_suite.py \
        --replace "return 0" "assert result.wasSuccessful(); return 0" \
        --replace "return 1" "assert result.wasSuccessful(); return 1"
    '';

    setupPyBuildFlags = with pkgs; [
      "--hdf5=${lib.getDev hdf5}"
      "--lzo=${lib.getDev lzo}"
      "--bzip2=${lib.getDev bzip2}"
      "--blosc=${lib.getDev c-blosc}"
    ];

    pythonImportsCheck = [ "tables" ];
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

  isort = super.isort.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [ self.poetry ];
  });

  entrypoints = super.entrypoints.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [ self.flit ];
  });

  pyyaml-env-tag = super.pyyaml-env-tag.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [ self.flit-core ];
  });

  questionary = super.questionary.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [ self.poetry ];
  });

  commitizen = super.commitizen.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [ self.poetry ];
  });

  testpath = super.testpath.overridePythonAttrs (attrs: {
    nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [ self.flit-core ];
  });
}
