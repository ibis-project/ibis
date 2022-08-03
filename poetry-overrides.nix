self: super:
let
  inherit (self) pkgs;
  inherit (pkgs) lib stdenv;
  numpyVersion = self.numpy.version;
  parallelizeSetupPy = drv: drv.overridePythonAttrs (attrs: {
    format = "setuptools";
    enableParallelBuilding = true;
    setupPyBuildFlags = attrs.setupPyBuildFlags or [ ] ++ [ "--parallel" "$NIX_BUILD_CORES" ];
  });
in
{
  # see https://github.com/numpy/numpy/issues/19624 for details
  numpy = super.numpy.overridePythonAttrs (attrs: {
    patches = (attrs.patches or [ ])
      ++ lib.optional
      # this patch only applies to macos and only with numpy versions >=1.21,<1.21.2
      (stdenv.isDarwin && (lib.versionAtLeast numpyVersion "1.21.0" && lib.versionOlder numpyVersion "1.21.2"))
      (pkgs.fetchpatch {
        url = "https://github.com/numpy/numpy/commit/8045183084042fbafc995dd26eb4d9ca45eb630e.patch";
        sha256 = "14g69vq7llkh6smpfrb50iqga7fd360dkcc0rlwb5k2cz8bsii5b";
      });
  });

  datafusion = super.datafusion.overridePythonAttrs (attrs: rec {
    inherit (attrs) version;
    src = pkgs.fetchFromGitHub {
      owner = "datafusion-contrib";
      repo = "datafusion-python";
      rev = attrs.version;
      sha256 = "sha256-9muPSFb4RjxP7X+qtUQ41rypgn0s9yWgmkyTA+edehU=";
    };

    patches = [
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

  tabulate = super.tabulate.overridePythonAttrs (_: {
    TABULATE_INSTALL = "lib-only";
  });

  pandas = parallelizeSetupPy super.pandas;
  pydantic = parallelizeSetupPy super.pydantic;

  mkdocstrings-python-legacy = super.mkdocstrings-python-legacy.overridePythonAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.pdm-pep517 ];
  });

  pkgutil-resolve-name = super.pkgutil-resolve-name.overrideAttrs (
    attrs: lib.optionalAttrs (lib.versionOlder self.python.version "3.9") {
      nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.flit-core ];
    }
  );

  watchdog = super.watchdog.overrideAttrs (attrs: lib.optionalAttrs
    (stdenv.isDarwin && lib.versionAtLeast attrs.version "2")
    {
      postPatch = ''
        substituteInPlace setup.py \
          --replace "if is_macos or os.getenv('FORCE_MACOS_MACHINE', '0') == '1':" 'if False:'
      '';
    });

  duckdb = super.duckdb.overrideAttrs (_: rec {
    inherit (pkgs.duckdb) version src patches;
    format = "setuptools";
    preConfigure = ''
      cd tools/pythonpkg
    '';
    SETUPTOOLS_SCM_PRETEND_VERSION = version;
  });

  adbc-driver-manager = self.buildPythonPackage {
    pname = "adbc-driver-manager";
    inherit (pkgs.adbc-driver-manager) version src;
    sourceRoot = "source/python/adbc_driver_manager";
    format = "setuptools";

    nativeBuildInputs = [ self.cython ];
    propagatedBuildInputs = [ self.pyarrow ];

    preCheck = ''
      export LD_LIBRARY_PATH="${pkgs.adbc-sqlite}/lib"
    '';

    checkInputs = [ self.pytest ];
    checkPhase = ''
      runHook preCheck
      pytest
      runHook postCheck
    '';
  };
}
