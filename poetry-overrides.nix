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
      owner = "apache";
      repo = "arrow-datafusion-python";
      rev = attrs.version;
      sha256 = "sha256-offZPEn+gL50ue36eonSQ3K4XNDrQItqorUI85CWVIE=";
    };

    patches = [ ./nix/patches/datafusion.patch ];

    cargoBuildNoDefaultFeatures = stdenv.isDarwin;
    nativeBuildInputs = attrs.nativeBuildInputs or [ ]
      ++ (with pkgs.rustPlatform; [ cargoSetupHook maturinBuildHook ]);

    buildInputs = attrs.buildInputs or [ ] ++ lib.optionals stdenv.isDarwin [
      pkgs.libiconv
      pkgs.darwin.apple_sdk.frameworks.Security
    ];

    cargoDeps = pkgs.rustPlatform.fetchCargoTarball {
      inherit src patches;
      sha256 = "sha256-mAVLZPQDmtXrh3nHfbIf1x5zWRWTInXKUC+pF41wCQY=";
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

  pyarrow = super.pyarrow.overridePythonAttrs (attrs: {
    PYARROW_WITH_DATASET = "1";
    PYARROW_WITH_FLIGHT = "1";
    PYARROW_WITH_HDFS = "0";
    PYARROW_WITH_PARQUET = "1";
    PYARROW_WITH_PLASMA = "0";
    PYARROW_WITH_S3 = "${if pkgs.arrow-cpp.enableS3 then "1" else "0"}";
    buildInputs = attrs.buildInputs or [ ] ++ lib.optionals (self.pythonOlder "3.9") [ pkgs.libxcrypt ];
  });

  mkdocs-material-extensions = super.mkdocs-material-extensions.overridePythonAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.hatchling ];
  });

  polars = super.polars.overridePythonAttrs (attrs:
    let
      inherit (attrs) version;
      src = pkgs.fetchFromGitHub {
        owner = "pola-rs";
        repo = "polars";
        rev = "py-${version}";
        sha256 = "sha256-cWhJRRXKbOipXXEpztEOrrJXG5WQcJc7cRjgiIx0XFQ=";
      };
      sourceRoot = "source/py-polars";
      nightlyRustPlatform = pkgs.makeRustPlatform {
        cargo = pkgs.rustNightly;
        rustc = pkgs.rustNightly;
      };
      patches = [ ./nix/patches/py-polars.patch ];
    in
    {
      inherit version src sourceRoot patches;

      nativeBuildInputs = attrs.nativeBuildInputs or [ ]
        ++ (with nightlyRustPlatform; [ cargoSetupHook maturinBuildHook ]);

      buildInputs = attrs.buildInputs or [ ]
        ++ lib.optionals stdenv.isDarwin [ pkgs.darwin.apple_sdk.frameworks.Security ];

      cargoDeps = nightlyRustPlatform.fetchCargoTarball {
        inherit src sourceRoot patches;
        name = "${attrs.pname}-${version}";
        sha256 = "sha256-fMeeYrSnCudU8PTMmoU2drlWluj+QiIQ+1DmiUb3AOo=";
      };
    });

  mkdocs-table-reader-plugin = super.mkdocs-table-reader-plugin.overridePythonAttrs (_: {
    postPatch = ''
      substituteInPlace setup.py --replace "tabulate>=0.8.7" "tabulate"
    '';
  });

  # hatch-requirements-txt has a huge set of (circular!) dependencies, only
  # used at build time none of which are required to build documentation so we
  # remove the part of pyproject.toml that requires it
  mkdocs-material = super.mkdocs-material.overridePythonAttrs (attrs: {
    postPatch = ''
      substituteInPlace pyproject.toml \
        --replace ', "hatch-requirements-txt"' "" \
        --replace '[tool.hatch.metadata.hooks.requirements_txt]' "" \
        --replace 'filename = "requirements.txt"' ""
    '';
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [
      self.hatchling
      self.hatch-nodejs-version
    ];
  });

  fiona = super.fiona.overridePythonAttrs (_: {
    format = "pyproject";
  });

  duckdb = super.duckdb.overridePythonAttrs (_: {
    postPatch = ''
      set -eo pipefail

      # fail if $NIX_BUILD_CORES is undefined
      set -u

      substituteInPlace setup.py \
        --replace 'multiprocessing.cpu_count()' $NIX_BUILD_CORES \
        --replace 'setuptools_scm<7.0.0' 'setuptools_scm'

      set +u
    '';
  });

  pymssql = super.pymssql.overridePythonAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.setuptools ];
    buildInputs = attrs.buildInputs or [ ] ++ [ pkgs.libkrb5 ];
  });

  nbclient = super.nbclient.overridePythonAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.hatchling ];
  });
}
