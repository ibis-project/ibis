final: prev: {
  pyspark = prev.pyspark.overridePythonAttrs (attrs:
    let
      inherit (final) pkgs lib;
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
      postInstall = attrs.postInstall or "" + ''
        cp ${icebergJar} $out/${final.python.sitePackages}/pyspark/jars/${icebergJar.name}
      '';
    });
}
