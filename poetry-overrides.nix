self: super: {
  # `wheel` cannot be used as a wheel to unpack itself, since that would
  # require itself (infinite recursion)
  wheel = super.wheel.override { preferWheel = false; };

  ipython-genutils = self.ipython_genutils;

  # TODO: Build grpcio from src on darwin because macOS 12_0 wheels won't install
  # TODO: Remove when the nix version of `pip` is upgraded?
  grpcio = super.grpcio.override { preferWheel = !self.pkgs.stdenv.isDarwin; };

  # No macOS 12 wheel available as of 2023-01-09
  debugpy = super.debugpy.override { preferWheel = !self.pkgs.stdenv.isDarwin; };
}
