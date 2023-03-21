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

  pydruid = super.pydruid.overridePythonAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.setuptools ];
  });

  ipython-genutils = self.ipython_genutils;

  mkdocs-jupyter =
    let
      linksPatch = self.pkgs.fetchpatch {
        name = "fix-mkdocs-jupyter-heading-links.patch";
        url = "https://github.com/danielfrg/mkdocs-jupyter/commit/f3b517580132fc743a34e5d9947731bc4f3c2143.patch";
        sha256 = "sha256-qcNobdcIziX3pFfnm6vxnhTqow/2VGI/+jbBs9jXkUo=";
      };
    in
    super.mkdocs-jupyter.overridePythonAttrs (_: {
      postFixup = ''
        cd $out/${self.python.sitePackages}
        patch -p1 < "${linksPatch}"
      '';
    });
} // super.lib.listToAttrs (
  map
    (name: {
      inherit name;
      value = super.${name}.override { preferWheel = !self.pkgs.stdenv.isDarwin; };
    })
    disabledMacOSWheels
)
