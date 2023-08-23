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
  paginate = super.paginate.overridePythonAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.setuptools ];
  });
  readtime = super.readtime.overridePythonAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.setuptools ];
  });
} // super.lib.listToAttrs (
  map
    (name: {
      inherit name;
      value = super.${name}.override { preferWheel = !self.pkgs.stdenv.isDarwin; };
    })
    disabledMacOSWheels
)
