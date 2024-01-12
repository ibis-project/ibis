SELECT
  "t32"."o_year",
  "t32"."mkt_share"
FROM (
  SELECT
    "t31"."o_year",
    CAST(SUM("t31"."nation_volume") AS DOUBLE) / SUM("t31"."volume") AS "mkt_share"
  FROM (
    SELECT
      "t30"."o_year",
      "t30"."volume",
      "t30"."nation",
      "t30"."r_name",
      "t30"."o_orderdate",
      "t30"."p_type",
      CASE WHEN "t30"."nation" = 'BRAZIL' THEN "t30"."volume" ELSE 0 END AS "nation_volume"
    FROM (
      SELECT
        EXTRACT(year FROM "t20"."o_orderdate") AS "o_year",
        "t18"."l_extendedprice" * (
          1 - "t18"."l_discount"
        ) AS "volume",
        "t22"."n_name" AS "nation",
        "t16"."r_name",
        "t20"."o_orderdate",
        "t17"."p_type"
      FROM (
        SELECT
          "t0"."p_partkey",
          "t0"."p_name",
          "t0"."p_mfgr",
          "t0"."p_brand",
          "t0"."p_type",
          "t0"."p_size",
          "t0"."p_container",
          CAST("t0"."p_retailprice" AS DECIMAL(15, 2)) AS "p_retailprice",
          "t0"."p_comment"
        FROM "hive"."ibis_sf1"."part" AS "t0"
      ) AS "t17"
      INNER JOIN (
        SELECT
          "t1"."l_orderkey",
          "t1"."l_partkey",
          "t1"."l_suppkey",
          "t1"."l_linenumber",
          CAST("t1"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
          CAST("t1"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
          CAST("t1"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
          CAST("t1"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
          "t1"."l_returnflag",
          "t1"."l_linestatus",
          "t1"."l_shipdate",
          "t1"."l_commitdate",
          "t1"."l_receiptdate",
          "t1"."l_shipinstruct",
          "t1"."l_shipmode",
          "t1"."l_comment"
        FROM "hive"."ibis_sf1"."lineitem" AS "t1"
      ) AS "t18"
        ON "t17"."p_partkey" = "t18"."l_partkey"
      INNER JOIN (
        SELECT
          "t2"."s_suppkey",
          "t2"."s_name",
          "t2"."s_address",
          "t2"."s_nationkey",
          "t2"."s_phone",
          CAST("t2"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
          "t2"."s_comment"
        FROM "hive"."ibis_sf1"."supplier" AS "t2"
      ) AS "t19"
        ON "t19"."s_suppkey" = "t18"."l_suppkey"
      INNER JOIN (
        SELECT
          "t3"."o_orderkey",
          "t3"."o_custkey",
          "t3"."o_orderstatus",
          CAST("t3"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t3"."o_orderdate",
          "t3"."o_orderpriority",
          "t3"."o_clerk",
          "t3"."o_shippriority",
          "t3"."o_comment"
        FROM "hive"."ibis_sf1"."orders" AS "t3"
      ) AS "t20"
        ON "t18"."l_orderkey" = "t20"."o_orderkey"
      INNER JOIN (
        SELECT
          "t4"."c_custkey",
          "t4"."c_name",
          "t4"."c_address",
          "t4"."c_nationkey",
          "t4"."c_phone",
          CAST("t4"."c_acctbal" AS DECIMAL(15, 2)) AS "c_acctbal",
          "t4"."c_mktsegment",
          "t4"."c_comment"
        FROM "hive"."ibis_sf1"."customer" AS "t4"
      ) AS "t21"
        ON "t20"."o_custkey" = "t21"."c_custkey"
      INNER JOIN (
        SELECT
          "t5"."n_nationkey",
          "t5"."n_name",
          "t5"."n_regionkey",
          "t5"."n_comment"
        FROM "hive"."ibis_sf1"."nation" AS "t5"
      ) AS "t14"
        ON "t21"."c_nationkey" = "t14"."n_nationkey"
      INNER JOIN (
        SELECT
          "t6"."r_regionkey",
          "t6"."r_name",
          "t6"."r_comment"
        FROM "hive"."ibis_sf1"."region" AS "t6"
      ) AS "t16"
        ON "t14"."n_regionkey" = "t16"."r_regionkey"
      INNER JOIN (
        SELECT
          "t5"."n_nationkey",
          "t5"."n_name",
          "t5"."n_regionkey",
          "t5"."n_comment"
        FROM "hive"."ibis_sf1"."nation" AS "t5"
      ) AS "t22"
        ON "t19"."s_nationkey" = "t22"."n_nationkey"
    ) AS "t30"
    WHERE
      "t30"."r_name" = 'AMERICA'
      AND "t30"."o_orderdate" BETWEEN FROM_ISO8601_DATE('1995-01-01') AND FROM_ISO8601_DATE('1996-12-31')
      AND "t30"."p_type" = 'ECONOMY ANODIZED STEEL'
  ) AS "t31"
  GROUP BY
    1
) AS "t32"
ORDER BY
  "t32"."o_year" ASC