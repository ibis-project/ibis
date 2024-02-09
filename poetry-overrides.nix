self: super: {
  # patch oscrypto for openssl 3 support: the fix is to relax some regexes that
  # inspect the libcrypto version
  #
  # without these patches applied, snowflake doesn't work in the nix environment
  #
  # these overrides can be removed when oscrypto is released again (1.3.0 was
  # released on 2022-03-17)
  oscrypto = (super.oscrypto.override { preferWheel = false; }).overridePythonAttrs (attrs: {
    patches = attrs.patches or [ ] ++ [
      (self.pkgs.fetchpatch {
        url = "https://github.com/wbond/oscrypto/commit/ebbc944485b278192b60080ea1f495e287efb4f8.patch";
        sha256 = "sha256-c1faM8szkn/7AjDthzmDisytzO8UdrzDtPkuuITjkRQ=";
      })
      (self.pkgs.fetchpatch {
        url = "https://github.com/wbond/oscrypto/commit/d5f3437ed24257895ae1edd9e503cfb352e635a8.patch";
        sha256 = "sha256-sRwxD99EV8mmiOAjM8emews9gvDeFtpBV3sSLiNEziM=";
      })
    ];
  });

  pyodbc = super.pyodbc.overridePythonAttrs (attrs: {
    preFixup = attrs.preFixup or "" + ''
      addAutoPatchelfSearchPath ${self.pkgs.unixODBC}
    '';
  });

  avro-python3 = super.avro-python3.overridePythonAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [
      self.pycodestyle
      self.isort
    ];
  });

  apache-flink-libraries = super.apache-flink-libraries.overridePythonAttrs (attrs: {
    buildInputs = attrs.nativeBuildInputs or [ ] ++ [ self.setuptools ];
    # apache-flink and apache-flink-libraries both install version.py into the
    # pyflink output derivation, which is invalid: whichever gets installed
    # last will be used
    postInstall = ''
      rm $out/${self.python.sitePackages}/pyflink/version.py
      rm $out/${self.python.sitePackages}/pyflink/__pycache__/version.*.pyc
    '';
  });
}
