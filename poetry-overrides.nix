self: super:
let
  disabledMacOSWheels = [
    "grpcio"
    "debugpy"
    "sqlalchemy"
    "greenlet"
  ];
in
{
  # `wheel` cannot be used as a wheel to unpack itself, since that would
  # require itself (infinite recursion)
  wheel = super.wheel.override { preferWheel = false; };
  cairocffi = (super.cairocffi.override {
    # FIXME(cpcloud): fix upstream in poetry2nix by ignoring patches when
    # preferWheel = true wheel can't be patched with upstream nixpkgs patch, so
    # build from source
    preferWheel = false;
  }).overridePythonAttrs (attrs: {
    # FIXME(cpcloud): this can be fixed upstream in poetry2nix by adding
    # flit-core to build-systems.json
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.flit-core ];
  });
} // super.lib.listToAttrs (
  map
    (name: {
      inherit name;
      value = super.${name}.override { preferWheel = !self.pkgs.stdenv.isDarwin; };
    })
    disabledMacOSWheels
)
