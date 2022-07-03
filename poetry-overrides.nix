{ lib, pkgs, stdenv, ... }:
let
  numpyVersion = self: self.numpy.version;
  parallelizeSetupPy = drv: drv.overridePythonAttrs (attrs: {
    format = "setuptools";
    enableParallelBuilding = true;
    setupPyBuildFlags = attrs.setupPyBuildFlags or [ ] ++ [ "--parallel" "$NIX_BUILD_CORES" ];
  });
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

  nbconvert = super.nbconvert.overridePythonAttrs (_: {
    postPatch = ''
      substituteInPlace \
        ./nbconvert/exporters/templateexporter.py \
        --replace \
        'root_dirs.extend(jupyter_path())' \
        'root_dirs.extend(jupyter_path() + [os.path.join("@out@", "share", "jupyter")])' \
        --subst-var out
    '';
  });

  tabulate = super.tabulate.overridePythonAttrs (_: {
    TABULATE_INSTALL = "lib-only";
  });

  pandas = parallelizeSetupPy super.pandas;
  pydantic = parallelizeSetupPy super.pydantic;

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
}
