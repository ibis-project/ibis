SELECT
  "t20"."nation",
  "t20"."o_year",
  "t20"."sum_profit"
FROM (
  SELECT
    "t19"."nation",
    "t19"."o_year",
    SUM("t19"."amount") AS "sum_profit"
  FROM (
    SELECT
      "t18"."amount",
      "t18"."o_year",
      "t18"."nation",
      "t18"."p_name"
    FROM (
      SELECT
        (
          "t13"."l_extendedprice" * (
            1 - "t13"."l_discount"
          )
        ) - (
          "t15"."ps_supplycost" * "t13"."l_quantity"
        ) AS "amount",
        EXTRACT(year FROM "t17"."o_orderdate") AS "o_year",
        "t12"."n_name" AS "nation",
        "t16"."p_name"
      FROM (
        SELECT
          "t0"."l_orderkey",
          "t0"."l_partkey",
          "t0"."l_suppkey",
          "t0"."l_linenumber",
          CAST("t0"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
          CAST("t0"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
          CAST("t0"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
          CAST("t0"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
          "t0"."l_returnflag",
          "t0"."l_linestatus",
          "t0"."l_shipdate",
          "t0"."l_commitdate",
          "t0"."l_receiptdate",
          "t0"."l_shipinstruct",
          "t0"."l_shipmode",
          "t0"."l_comment"
        FROM "hive"."ibis_sf1"."lineitem" AS "t0"
      ) AS "t13"
      INNER JOIN (
        SELECT
          "t1"."s_suppkey",
          "t1"."s_name",
          "t1"."s_address",
          "t1"."s_nationkey",
          "t1"."s_phone",
          CAST("t1"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
          "t1"."s_comment"
        FROM "hive"."ibis_sf1"."supplier" AS "t1"
      ) AS "t14"
        ON "t14"."s_suppkey" = "t13"."l_suppkey"
      INNER JOIN (
        SELECT
          "t2"."ps_partkey",
          "t2"."ps_suppkey",
          "t2"."ps_availqty",
          CAST("t2"."ps_supplycost" AS DECIMAL(15, 2)) AS "ps_supplycost",
          "t2"."ps_comment"
        FROM "hive"."ibis_sf1"."partsupp" AS "t2"
      ) AS "t15"
        ON "t15"."ps_suppkey" = "t13"."l_suppkey" AND "t15"."ps_partkey" = "t13"."l_partkey"
      INNER JOIN (
        SELECT
          "t3"."p_partkey",
          "t3"."p_name",
          "t3"."p_mfgr",
          "t3"."p_brand",
          "t3"."p_type",
          "t3"."p_size",
          "t3"."p_container",
          CAST("t3"."p_retailprice" AS DECIMAL(15, 2)) AS "p_retailprice",
          "t3"."p_comment"
        FROM "hive"."ibis_sf1"."part" AS "t3"
      ) AS "t16"
        ON "t16"."p_partkey" = "t13"."l_partkey"
      INNER JOIN (
        SELECT
          "t4"."o_orderkey",
          "t4"."o_custkey",
          "t4"."o_orderstatus",
          CAST("t4"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t4"."o_orderdate",
          "t4"."o_orderpriority",
          "t4"."o_clerk",
          "t4"."o_shippriority",
          "t4"."o_comment"
        FROM "hive"."ibis_sf1"."orders" AS "t4"
      ) AS "t17"
        ON "t17"."o_orderkey" = "t13"."l_orderkey"
      INNER JOIN (
        SELECT
          "t5"."n_nationkey",
          "t5"."n_name",
          "t5"."n_regionkey",
          "t5"."n_comment"
        FROM "hive"."ibis_sf1"."nation" AS "t5"
      ) AS "t12"
        ON "t14"."s_nationkey" = "t12"."n_nationkey"
    ) AS "t18"
    WHERE
      "t18"."p_name" LIKE '%green%'
  ) AS "t19"
  GROUP BY
    1,
    2
) AS "t20"
ORDER BY
  "t20"."nation" ASC,
  "t20"."o_year" DESC