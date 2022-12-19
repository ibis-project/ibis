self: super: {
  mkdocstrings = super.mkdocstrings.overridePythonAttrs (_: {
    # patch the installed mkdocstrings sources to fix jinja2 imports
    #
    # strip the first two levels "a/src/" when patching since we're in site-packages
    # just above mkdocstrings
    postInstall = ''
      pushd "$out/${self.python.sitePackages}"
      patch -p2 < "${self.pkgs.fetchpatch {
        name = "fix-jinja2-imports.patch";
        url = "https://github.com/mkdocstrings/mkdocstrings/commit/b37722716b1e0ed6393ec71308dfb0f85e142f3b.patch";
        hash = "sha256-DD1SjEvs5HBlSRLrqP3jhF/yoeWkF7F3VXCD1gyt5Fc=";
      }}"
      popd
    '';
  });

  # `wheel` cannot be used as a wheel to unpack itself, since that would
  # require itself (infinite recursion)
  wheel = super.wheel.override { preferWheel = false; };

  ipython-genutils = self.ipython_genutils;

  # without this we won't pick up the necessary patches to nbconvert's paths to
  # jupyter in the nix store
  nbconvert = super.nbconvert.override { preferWheel = false; };

  # TODO: Build grpcio from src on darwin because macOS 12_0 wheels won't install
  # TODO: Remove when the nix version of `pip` is upgraded?
  grpcio = super.grpcio.override { preferWheel = !self.pkgs.stdenv.isDarwin; };
}
