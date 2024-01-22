SELECT
  "t20"."n_name",
  "t20"."revenue"
FROM (
  SELECT
    "t19"."n_name",
    SUM("t19"."l_extendedprice" * (
      1 - "t19"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t18"."c_custkey",
      "t18"."c_name",
      "t18"."c_address",
      "t18"."c_nationkey",
      "t18"."c_phone",
      "t18"."c_acctbal",
      "t18"."c_mktsegment",
      "t18"."c_comment",
      "t18"."o_orderkey",
      "t18"."o_custkey",
      "t18"."o_orderstatus",
      "t18"."o_totalprice",
      "t18"."o_orderdate",
      "t18"."o_orderpriority",
      "t18"."o_clerk",
      "t18"."o_shippriority",
      "t18"."o_comment",
      "t18"."l_orderkey",
      "t18"."l_partkey",
      "t18"."l_suppkey",
      "t18"."l_linenumber",
      "t18"."l_quantity",
      "t18"."l_extendedprice",
      "t18"."l_discount",
      "t18"."l_tax",
      "t18"."l_returnflag",
      "t18"."l_linestatus",
      "t18"."l_shipdate",
      "t18"."l_commitdate",
      "t18"."l_receiptdate",
      "t18"."l_shipinstruct",
      "t18"."l_shipmode",
      "t18"."l_comment",
      "t18"."s_suppkey",
      "t18"."s_name",
      "t18"."s_address",
      "t18"."s_nationkey",
      "t18"."s_phone",
      "t18"."s_acctbal",
      "t18"."s_comment",
      "t18"."n_nationkey",
      "t18"."n_name",
      "t18"."n_regionkey",
      "t18"."n_comment",
      "t18"."r_regionkey",
      "t18"."r_name",
      "t18"."r_comment"
    FROM (
      SELECT
        "t14"."c_custkey",
        "t14"."c_name",
        "t14"."c_address",
        "t14"."c_nationkey",
        "t14"."c_phone",
        "t14"."c_acctbal",
        "t14"."c_mktsegment",
        "t14"."c_comment",
        "t15"."o_orderkey",
        "t15"."o_custkey",
        "t15"."o_orderstatus",
        "t15"."o_totalprice",
        "t15"."o_orderdate",
        "t15"."o_orderpriority",
        "t15"."o_clerk",
        "t15"."o_shippriority",
        "t15"."o_comment",
        "t16"."l_orderkey",
        "t16"."l_partkey",
        "t16"."l_suppkey",
        "t16"."l_linenumber",
        "t16"."l_quantity",
        "t16"."l_extendedprice",
        "t16"."l_discount",
        "t16"."l_tax",
        "t16"."l_returnflag",
        "t16"."l_linestatus",
        "t16"."l_shipdate",
        "t16"."l_commitdate",
        "t16"."l_receiptdate",
        "t16"."l_shipinstruct",
        "t16"."l_shipmode",
        "t16"."l_comment",
        "t17"."s_suppkey",
        "t17"."s_name",
        "t17"."s_address",
        "t17"."s_nationkey",
        "t17"."s_phone",
        "t17"."s_acctbal",
        "t17"."s_comment",
        "t12"."n_nationkey",
        "t12"."n_name",
        "t12"."n_regionkey",
        "t12"."n_comment",
        "t13"."r_regionkey",
        "t13"."r_name",
        "t13"."r_comment"
      FROM (
        SELECT
          "t0"."c_custkey",
          "t0"."c_name",
          "t0"."c_address",
          "t0"."c_nationkey",
          "t0"."c_phone",
          CAST("t0"."c_acctbal" AS DECIMAL(15, 2)) AS "c_acctbal",
          "t0"."c_mktsegment",
          "t0"."c_comment"
        FROM "hive"."ibis_sf1"."customer" AS "t0"
      ) AS "t14"
      INNER JOIN (
        SELECT
          "t1"."o_orderkey",
          "t1"."o_custkey",
          "t1"."o_orderstatus",
          CAST("t1"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t1"."o_orderdate",
          "t1"."o_orderpriority",
          "t1"."o_clerk",
          "t1"."o_shippriority",
          "t1"."o_comment"
        FROM "hive"."ibis_sf1"."orders" AS "t1"
      ) AS "t15"
        ON "t14"."c_custkey" = "t15"."o_custkey"
      INNER JOIN (
        SELECT
          "t2"."l_orderkey",
          "t2"."l_partkey",
          "t2"."l_suppkey",
          "t2"."l_linenumber",
          CAST("t2"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
          CAST("t2"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
          CAST("t2"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
          CAST("t2"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
          "t2"."l_returnflag",
          "t2"."l_linestatus",
          "t2"."l_shipdate",
          "t2"."l_commitdate",
          "t2"."l_receiptdate",
          "t2"."l_shipinstruct",
          "t2"."l_shipmode",
          "t2"."l_comment"
        FROM "hive"."ibis_sf1"."lineitem" AS "t2"
      ) AS "t16"
        ON "t16"."l_orderkey" = "t15"."o_orderkey"
      INNER JOIN (
        SELECT
          "t3"."s_suppkey",
          "t3"."s_name",
          "t3"."s_address",
          "t3"."s_nationkey",
          "t3"."s_phone",
          CAST("t3"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
          "t3"."s_comment"
        FROM "hive"."ibis_sf1"."supplier" AS "t3"
      ) AS "t17"
        ON "t16"."l_suppkey" = "t17"."s_suppkey"
      INNER JOIN (
        SELECT
          "t4"."n_nationkey",
          "t4"."n_name",
          "t4"."n_regionkey",
          "t4"."n_comment"
        FROM "hive"."ibis_sf1"."nation" AS "t4"
      ) AS "t12"
        ON "t14"."c_nationkey" = "t17"."s_nationkey"
        AND "t17"."s_nationkey" = "t12"."n_nationkey"
      INNER JOIN (
        SELECT
          "t5"."r_regionkey",
          "t5"."r_name",
          "t5"."r_comment"
        FROM "hive"."ibis_sf1"."region" AS "t5"
      ) AS "t13"
        ON "t12"."n_regionkey" = "t13"."r_regionkey"
    ) AS "t18"
    WHERE
      "t18"."r_name" = 'ASIA'
      AND "t18"."o_orderdate" >= FROM_ISO8601_DATE('1994-01-01')
      AND "t18"."o_orderdate" < FROM_ISO8601_DATE('1995-01-01')
  ) AS "t19"
  GROUP BY
    1
) AS "t20"
ORDER BY
  "t20"."revenue" DESC