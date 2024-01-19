WITH "t8" AS (
  SELECT
    "t6"."n_nationkey",
    "t6"."n_name",
    "t6"."n_regionkey",
    "t6"."n_comment"
  FROM "hive"."ibis_sf1"."nation" AS "t6"
)
SELECT
  "t33"."o_year",
  "t33"."mkt_share"
FROM (
  SELECT
    "t32"."o_year",
    CAST(SUM("t32"."nation_volume") AS DOUBLE) / SUM("t32"."volume") AS "mkt_share"
  FROM (
    SELECT
      "t31"."o_year",
      "t31"."volume",
      "t31"."nation",
      "t31"."r_name",
      "t31"."o_orderdate",
      "t31"."p_type",
      CASE WHEN "t31"."nation" = 'BRAZIL' THEN "t31"."volume" ELSE 0 END AS "nation_volume"
    FROM (
      SELECT
        EXTRACT(year FROM "t19"."o_orderdate") AS "o_year",
        "t17"."l_extendedprice" * (
          1 - "t17"."l_discount"
        ) AS "volume",
        "t23"."n_name" AS "nation",
        "t14"."r_name",
        "t19"."o_orderdate",
        "t16"."p_type"
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
      ) AS "t16"
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
      ) AS "t17"
        ON "t16"."p_partkey" = "t17"."l_partkey"
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
      ) AS "t18"
        ON "t18"."s_suppkey" = "t17"."l_suppkey"
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
      ) AS "t19"
        ON "t17"."l_orderkey" = "t19"."o_orderkey"
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
      ) AS "t20"
        ON "t19"."o_custkey" = "t20"."c_custkey"
      INNER JOIN "t8" AS "t21"
        ON "t20"."c_nationkey" = "t21"."n_nationkey"
      INNER JOIN (
        SELECT
          "t5"."r_regionkey",
          "t5"."r_name",
          "t5"."r_comment"
        FROM "hive"."ibis_sf1"."region" AS "t5"
      ) AS "t14"
        ON "t21"."n_regionkey" = "t14"."r_regionkey"
      INNER JOIN "t8" AS "t23"
        ON "t18"."s_nationkey" = "t23"."n_nationkey"
    ) AS "t31"
    WHERE
      "t31"."r_name" = 'AMERICA'
      AND "t31"."o_orderdate" BETWEEN FROM_ISO8601_DATE('1995-01-01') AND FROM_ISO8601_DATE('1996-12-31')
      AND "t31"."p_type" = 'ECONOMY ANODIZED STEEL'
  ) AS "t32"
  GROUP BY
    1
) AS "t33"
ORDER BY
  "t33"."o_year" ASC