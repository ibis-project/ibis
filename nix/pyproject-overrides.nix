{ pkgs }: final: prev:
let
  inherit (pkgs) lib stdenv;
  inherit (final) resolveBuildSystem;

  addBuildSystems =
    pkg: spec:
    pkg.overrideAttrs (old: {
      nativeBuildInputs = old.nativeBuildInputs ++ resolveBuildSystem spec;
    });

  buildSystemOverrides = {
    atpublic.hatchling = [ ];
    packaging.flit-core = [ ];
    parsy.setuptools = [ ];
    pathspec.flit-core = [ ];
    pluggy = {
      setuptools = [ ];
      setuptools-scm = [ ];
    };
    pure-sasl.setuptools = [ ];
    pydruid.setuptools = [ ];
    pytest-clarity.setuptools = [ ];
    sqlglot = {
      setuptools = [ ];
      setuptools-scm = [ ];
    };
    psycopg2.setuptools = [ ];
    tomli.flit-core = [ ];
    toolz.setuptools = [ ];
    typing-extensions.flit-core = [ ];
    debugpy.setuptools = [ ];
    google-crc32c.setuptools = [ ];
    lz4.setuptools = [ ];
    snowflake-connector-python.setuptools = [ ];
  } // lib.optionalAttrs stdenv.hostPlatform.isDarwin {
    duckdb = {
      setuptools = [ ];
      setuptools-scm = [ ];
      pybind11 = [ ];
    };
  };
in
(lib.optionalAttrs stdenv.hostPlatform.isDarwin {
  pyproj = prev.pyproj.overrideAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [
      final.setuptools
      final.cython
      pkgs.proj
    ];
    PROJ_DIR = "${lib.getBin pkgs.proj}";
    PROJ_INCDIR = "${lib.getDev pkgs.proj}";
  });

  psygnal = prev.psygnal.overrideAttrs (_: {
    src = pkgs.fetchFromGitHub {
      owner = "pyapp-kit";
      repo = prev.psygnal.pname;
      rev = "refs/tags/v${prev.psygnal.version}";
      hash = "sha256-eGJWtmw2Ps3jII4T8E6s3djzxfqcSdyPemvejal0cn4=";
    };
  });
}) // lib.mapAttrs (name: spec: addBuildSystems prev.${name} spec) buildSystemOverrides // {
  mysqlclient = prev.mysqlclient.overrideAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ final.setuptools ];
    buildInputs = attrs.buildInputs or [ ] ++ [ pkgs.pkg-config pkgs.libmysqlclient ];
  });

  psycopg2 = prev.psycopg2.overrideAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ final.setuptools ];
    buildInputs = [ pkgs.postgresql ]
    ++ lib.optionals stdenv.hostPlatform.isDarwin [ pkgs.openssl ];
  });

  pyodbc = prev.pyodbc.overrideAttrs (attrs: {
    buildInputs = attrs.buildInputs or [ ] ++ [ pkgs.unixODBC ];
  });

  pyspark = prev.pyspark.overrideAttrs (attrs:
    let
      pysparkVersion = lib.versions.majorMinor attrs.version;
      jarHashes = {
        "3.5" = "sha256-h+cYTzHvDKrEFbvfzxvElDNGpYuY10fcg0NPcTnhKss=";
        "3.3" = "sha256-3D++9VCiLoMP7jPvdCtBn7xnxqHnyQowcqdGUe0M3mk=";
      };
      icebergVersion = "1.6.1";
      scalaVersion = "2.12";
      jarName = "iceberg-spark-runtime-${pysparkVersion}_${scalaVersion}-${icebergVersion}.jar";
      icebergJarUrl = "https://search.maven.org/remotecontent?filepath=org/apache/iceberg/iceberg-spark-runtime-${pysparkVersion}_${scalaVersion}/${icebergVersion}/${jarName}";
      icebergJar = pkgs.fetchurl {
        name = jarName;
        url = icebergJarUrl;
        sha256 = jarHashes."${pysparkVersion}";
      };
    in
    {
      nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ final.setuptools ];
      postInstall = attrs.postInstall or "" + ''
        cp ${icebergJar} $out/${final.python.sitePackages}/pyspark/jars/${icebergJar.name}
      '';
    });

  thrift = prev.thrift.overrideAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ final.setuptools ];
    # avoid extremely premature optimization so that we don't have to
    # deal with a useless dependency on distutils
    postPatch = attrs.postPatch or "" + lib.optionalString (final.python.pythonAtLeast "3.12") ''
      substituteInPlace setup.cfg --replace 'optimize = 1' 'optimize = 0'
    '';
  });
}
