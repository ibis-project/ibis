final: prev: {
  pyspark = prev.pyspark.overridePythonAttrs (attrs:
    let
      icebergJarUrl = "https://search.maven.org/remotecontent?filepath=org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.5.2/iceberg-spark-runtime-3.5_2.12-1.5.2.jar";
      icebergJar = final.pkgs.fetchurl {
        name = "iceberg-spark-runtime-3.5_2.12-1.5.2.jar";
        url = icebergJarUrl;
        sha256 = "12v1704h0bq3qr2fci0mckg9171lyr8v6983wpa83k06v1w4pv1a";
      };
    in
    {
      postInstall = attrs.postInstall or "" + ''
        cp ${icebergJar} $out/${final.python.sitePackages}/pyspark/jars/${icebergJar.name}
      '';
    });
}
