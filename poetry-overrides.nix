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

  typing-extensions = super.typing-extensions.overridePythonAttrs (attrs: {
    buildInputs = (attrs.buildInputs or [ ]) ++ [ self.flit-core ];
  });
}
