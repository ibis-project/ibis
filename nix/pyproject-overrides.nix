{ pkgs }:
final: prev:
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
    pandas-gbq.setuptools = [ ];
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
    tomli.flit-core = [ ];
    toolz.setuptools = [ ];
    typing-extensions.flit-core = [ ];
    debugpy.setuptools = [ ];
    google-crc32c.setuptools = [ ];
    lz4.setuptools = [ ];
    snowflake-connector-python.setuptools = [ ];
  }
  // lib.optionalAttrs (lib.versionAtLeast prev.python.pythonVersion "3.13") {
    pyyaml-ft.setuptools = [ ];
  }
  // lib.optionalAttrs stdenv.hostPlatform.isDarwin {
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

  watchdog = prev.watchdog.overrideAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [
      final.setuptools
    ];
  });
})
// lib.mapAttrs (name: spec: addBuildSystems prev.${name} spec) buildSystemOverrides
// {
  hatchling = prev.hatchling.overrideAttrs (attrs: {
    propagatedBuildInputs = attrs.propagatedBuildInputs or [ ] ++ [ final.editables ];
  });

  # pandas python 3.10 wheels on manylinux aarch64 somehow ships shared objects
  # for all versions of python
  pandas = prev.pandas.overrideAttrs (
    attrs:
    let
      py = final.python;
      shortVersion = lib.replaceStrings [ "." ] [ "" ] py.pythonVersion;
      impl = py.implementation;
    in
    lib.optionalAttrs (stdenv.isAarch64 && stdenv.isLinux && shortVersion == "310") {
      postInstall = attrs.postInstall or "" + ''
        find $out \
          \( -name '*.${impl}-*.so' -o -name 'libgcc*' -o -name 'libstdc*' \) \
          -a ! -name '*.${impl}-${shortVersion}-*.so' \
          -delete
      '';
    }
  );

  psygnal = prev.psygnal.overrideAttrs (
    attrs:
    {
      nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [
        final.hatchling
        final.pathspec
        final.pluggy
        final.packaging
        final.trove-classifiers
      ];
    }
    // lib.optionalAttrs stdenv.hostPlatform.isDarwin {
      src = pkgs.fetchFromGitHub {
        owner = "pyapp-kit";
        repo = prev.psygnal.pname;
        rev = "refs/tags/v${prev.psygnal.version}";
        hash = "sha256-eGJWtmw2Ps3jII4T8E6s3djzxfqcSdyPemvejal0cn4=";
      };
    }
  );

  mysqlclient = prev.mysqlclient.overrideAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ final.setuptools ];
    buildInputs = attrs.buildInputs or [ ] ++ [
      pkgs.pkg-config
      pkgs.libmysqlclient
    ];
  });

  psycopg2 = prev.psycopg2.overrideAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ final.setuptools ];
    buildInputs =
      attrs.buildInputs or [ ]
      ++ [ pkgs.libpq.pg_config ]
      ++ lib.optionals stdenv.hostPlatform.isDarwin [ pkgs.openssl ];
  });

  pyodbc = prev.pyodbc.overrideAttrs (attrs: {
    buildInputs = attrs.buildInputs or [ ] ++ [ pkgs.unixODBC ];
  });

  pyspark = prev.pyspark.overrideAttrs (
    attrs:
    let
      pysparkVersion = lib.versions.majorMinor attrs.version;
      icebergVersion = "1.10.1";
      jarSpecs = {
        "4.0" = {
          scalaVersion = "2.13";
          hash = "sha256-IZKgiB7Q9Xc7WoOogg0rCyBpvuwgMChkOhwzhVEAfwk=";
        };
        "3.5" = {
          scalaVersion = "2.12";
          hash = "sha256-OeoJ5sA1UKMAudmrSYlJg20tQ05EHaTBKpQ6CG45aUA=";
        };
      };
      jarInfo = jarSpecs."${pysparkVersion}";
      inherit (jarInfo) scalaVersion;
      jarName = "iceberg-spark-runtime-${pysparkVersion}_${scalaVersion}-${icebergVersion}.jar";
      icebergJarUrl = "https://search.maven.org/remotecontent?filepath=org/apache/iceberg/iceberg-spark-runtime-${pysparkVersion}_${scalaVersion}/${icebergVersion}/${jarName}";
      icebergJar = pkgs.fetchurl {
        name = jarName;
        url = icebergJarUrl;
        sha256 = jarInfo.hash;
      };
    in
    {
      nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ final.setuptools ];
      postInstall = attrs.postInstall or "" + ''
        sitePackagesPySpark=$out/${final.python.sitePackages}/pyspark
        cp -v ${icebergJar} $sitePackagesPySpark/jars/${icebergJar.name}
        mkdir -p $sitePackagesPySpark/conf
        cp -v ${../docker/spark-connect/log4j2.properties} $sitePackagesPySpark/conf/log4j2.properties
      '';
    }
  );

  thrift = prev.thrift.overrideAttrs (attrs: {
    nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ final.setuptools ];
    # avoid extremely premature optimization so that we don't have to
    # deal with a useless dependency on distutils
    postPatch =
      attrs.postPatch or ""
      + lib.optionalString (final.python.pythonAtLeast "3.12") ''
        substituteInPlace setup.cfg --replace 'optimize = 1' 'optimize = 0'
      '';
  });
}
