SELECT
  "t14"."c_custkey",
  "t14"."c_name",
  "t14"."revenue",
  "t14"."c_acctbal",
  "t14"."n_name",
  "t14"."c_address",
  "t14"."c_phone",
  "t14"."c_comment"
FROM (
  SELECT
    "t13"."c_custkey",
    "t13"."c_name",
    "t13"."c_acctbal",
    "t13"."n_name",
    "t13"."c_address",
    "t13"."c_phone",
    "t13"."c_comment",
    SUM("t13"."l_extendedprice" * (
      1 - "t13"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t12"."c_custkey",
      "t12"."c_name",
      "t12"."c_address",
      "t12"."c_nationkey",
      "t12"."c_phone",
      "t12"."c_acctbal",
      "t12"."c_mktsegment",
      "t12"."c_comment",
      "t12"."o_orderkey",
      "t12"."o_custkey",
      "t12"."o_orderstatus",
      "t12"."o_totalprice",
      "t12"."o_orderdate",
      "t12"."o_orderpriority",
      "t12"."o_clerk",
      "t12"."o_shippriority",
      "t12"."o_comment",
      "t12"."l_orderkey",
      "t12"."l_partkey",
      "t12"."l_suppkey",
      "t12"."l_linenumber",
      "t12"."l_quantity",
      "t12"."l_extendedprice",
      "t12"."l_discount",
      "t12"."l_tax",
      "t12"."l_returnflag",
      "t12"."l_linestatus",
      "t12"."l_shipdate",
      "t12"."l_commitdate",
      "t12"."l_receiptdate",
      "t12"."l_shipinstruct",
      "t12"."l_shipmode",
      "t12"."l_comment",
      "t12"."n_nationkey",
      "t12"."n_name",
      "t12"."n_regionkey",
      "t12"."n_comment"
    FROM (
      SELECT
        "t9"."c_custkey",
        "t9"."c_name",
        "t9"."c_address",
        "t9"."c_nationkey",
        "t9"."c_phone",
        "t9"."c_acctbal",
        "t9"."c_mktsegment",
        "t9"."c_comment",
        "t10"."o_orderkey",
        "t10"."o_custkey",
        "t10"."o_orderstatus",
        "t10"."o_totalprice",
        "t10"."o_orderdate",
        "t10"."o_orderpriority",
        "t10"."o_clerk",
        "t10"."o_shippriority",
        "t10"."o_comment",
        "t11"."l_orderkey",
        "t11"."l_partkey",
        "t11"."l_suppkey",
        "t11"."l_linenumber",
        "t11"."l_quantity",
        "t11"."l_extendedprice",
        "t11"."l_discount",
        "t11"."l_tax",
        "t11"."l_returnflag",
        "t11"."l_linestatus",
        "t11"."l_shipdate",
        "t11"."l_commitdate",
        "t11"."l_receiptdate",
        "t11"."l_shipinstruct",
        "t11"."l_shipmode",
        "t11"."l_comment",
        "t8"."n_nationkey",
        "t8"."n_name",
        "t8"."n_regionkey",
        "t8"."n_comment"
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
      ) AS "t9"
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
      ) AS "t10"
        ON "t9"."c_custkey" = "t10"."o_custkey"
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
      ) AS "t11"
        ON "t11"."l_orderkey" = "t10"."o_orderkey"
      INNER JOIN (
        SELECT
          "t3"."n_nationkey",
          "t3"."n_name",
          "t3"."n_regionkey",
          "t3"."n_comment"
        FROM "hive"."ibis_sf1"."nation" AS "t3"
      ) AS "t8"
        ON "t9"."c_nationkey" = "t8"."n_nationkey"
    ) AS "t12"
    WHERE
      "t12"."o_orderdate" >= FROM_ISO8601_DATE('1993-10-01')
      AND "t12"."o_orderdate" < FROM_ISO8601_DATE('1994-01-01')
      AND "t12"."l_returnflag" = 'R'
  ) AS "t13"
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
) AS "t14"
ORDER BY
  "t14"."revenue" DESC
LIMIT 20