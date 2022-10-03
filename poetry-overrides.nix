self: super:
let
  inherit (self) pkgs;
  inherit (pkgs) lib stdenv;

  numpyVersion = super.numpy.version;
  versionBetween = version: lower: upper:
    lib.versionAtLeast version lower && lib.versionOlder version upper;
in
{
  # this patch only applies to macos and only with numpy versions >=1.21,<1.21.2
  # see https://github.com/numpy/numpy/issues/19624 for details
  numpy = super.numpy.overridePythonAttrs (
    attrs: lib.optionalAttrs
      (stdenv.isDarwin && versionBetween numpyVersion "1.21.0" "1.21.2")
      {
        patches = attrs.patches or [ ] ++ [
          (pkgs.fetchpatch {
            url = "https://github.com/numpy/numpy/commit/8045183084042fbafc995dd26eb4d9ca45eb630e.patch";
            sha256 = "14g69vq7llkh6smpfrb50iqga7fd360dkcc0rlwb5k2cz8bsii5b";
          })
        ];
      } // lib.optionalAttrs (lib.versionAtLeast numpyVersion "1.23.3") {
      format = "setuptools";
    }
  );

  datafusion = super.datafusion.overridePythonAttrs (attrs: rec {
    inherit (attrs) version;
    src = pkgs.fetchFromGitHub {
      owner = "datafusion-contrib";
      repo = "datafusion-python";
      rev = attrs.version;
      sha256 = "sha256-9muPSFb4RjxP7X+qtUQ41rypgn0s9yWgmkyTA+edehU=";
    };

    patches = attrs.patches or [ ] ++ [
      (pkgs.fetchpatch {
        name = "optional-mimalloc.patch";
        url = "https://github.com/datafusion-contrib/datafusion-python/commit/a5b10e8ef19514361fc6062a8ad63d7a793c2111.patch";
        sha256 = "sha256-vmB1FKb2VeecrQt91J+pDp+2jvdtOrGd4w4wjhDMJK8=";
      })
    ];

    cargoBuildNoDefaultFeatures = stdenv.isDarwin;
    nativeBuildInputs = attrs.nativeBuildInputs or [ ]
      ++ (with pkgs.rustPlatform; [ cargoSetupHook maturinBuildHook ]);

    buildInputs = attrs.buildInputs or [ ]
      ++ lib.optionals stdenv.isDarwin [ pkgs.libiconv ];

    cargoDeps = pkgs.rustPlatform.fetchCargoTarball {
      inherit src patches;
      sha256 = "sha256-rGXSmn3MF2wFyMqzF15gB9DK5f9W4Gk08J7tOsZ7IH0=";
    };
  });

  pandas = super.pandas.overridePythonAttrs (attrs: {
    format = "setuptools";
    enableParallelBuilding = true;
    setupPyBuildFlags = attrs.setupPyBuildFlags or [ ] ++ [ "--parallel" "$NIX_BUILD_CORES" ];
  });

  mkdocstrings = super.mkdocstrings.overridePythonAttrs (attrs: {
    patches = attrs.patches or [ ] ++ [
      (pkgs.fetchpatch {
        name = "fix-jinja2-imports.patch";
        url = "https://github.com/mkdocstrings/mkdocstrings/commit/b37722716b1e0ed6393ec71308dfb0f85e142f3b.patch";
        sha256 = "sha256-DD1SjEvs5HBlSRLrqP3jhF/yoeWkF7F3VXCD1gyt5Fc=";
      })
    ];
  });

  watchdog = super.watchdog.overrideAttrs (attrs: lib.optionalAttrs
    (stdenv.isDarwin && lib.versionAtLeast attrs.version "2")
    {
      postPatch = ''
        substituteInPlace setup.py \
          --replace "if is_macos or os.getenv('FORCE_MACOS_MACHINE', '0') == '1':" 'if False:'
      '';
    });

  pyarrow = super.pyarrow.overridePythonAttrs (_: {
    PYARROW_WITH_DATASET = "1";
    PYARROW_WITH_FLIGHT = "1";
    PYARROW_WITH_HDFS = "0";
    PYARROW_WITH_PARQUET = "1";
    PYARROW_WITH_PLASMA = "0";
    PYARROW_WITH_S3 = "${if pkgs.arrow-cpp.enableS3 then "1" else "0"}";
  });

  polars = super.polars.overridePythonAttrs (attrs:
    let
      inherit (attrs) version;
      src = pkgs.fetchFromGitHub {
        owner = "pola-rs";
        repo = "polars";
        rev = "py-${version}";
        sha256 = "sha256-u4mBAnX8pG2kb6ywlG+jkUpYG5Z/vvZdQSfz2pwQT6A=";
      };
      sourceRoot = "source/py-polars";
      nightlyRustPlatform = pkgs.makeRustPlatform {
        cargo = pkgs.rustNightly;
        rustc = pkgs.rustNightly;
      };
    in
    {
      inherit version src sourceRoot;

      patches = [ ./nix/patches/py-polars.patch ];

      nativeBuildInputs = attrs.nativeBuildInputs or [ ]
        ++ (with nightlyRustPlatform; [ cargoSetupHook maturinBuildHook ]);

      cargoDeps = nightlyRustPlatform.fetchCargoTarball {
        inherit src sourceRoot;
        patches = [ ./nix/patches/py-polars.patch ];
        name = "${attrs.pname}-${version}";
        sha256 = "sha256-aQIWSDwwGYpJMUcUIYd68G7yFv87QwdvYevbeolTj88=";
      };
    });

  mkdocs-table-reader-plugin = super.mkdocs-table-reader-plugin.overridePythonAttrs (attrs: {
    propagatedBuildInputs = attrs.propagatedBuildInputs or [ ] ++ [ self.tabulate ];
    postPatch = ''
      substituteInPlace setup.py --replace "tabulate>=0.8.7" "tabulate"
    '';
  });

  fiona = super.fiona.overridePythonAttrs (_: {
    format = "setuptools";
  });

  backports-zoneinfo = super.backports-zoneinfo.overridePythonAttrs (attrs:
    lib.optionalAttrs (self.pythonOlder "3.9") {
      nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.setuptools ];
    }
  );
}
