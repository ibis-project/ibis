SELECT
  "t11"."l_orderkey",
  "t11"."revenue",
  "t11"."o_orderdate",
  "t11"."o_shippriority"
FROM (
  SELECT
    "t10"."l_orderkey",
    "t10"."o_orderdate",
    "t10"."o_shippriority",
    SUM("t10"."l_extendedprice" * (
      1 - "t10"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      *
    FROM (
      SELECT
        "t6"."c_custkey",
        "t6"."c_name",
        "t6"."c_address",
        "t6"."c_nationkey",
        "t6"."c_phone",
        "t6"."c_acctbal",
        "t6"."c_mktsegment",
        "t6"."c_comment",
        "t7"."o_orderkey",
        "t7"."o_custkey",
        "t7"."o_orderstatus",
        "t7"."o_totalprice",
        "t7"."o_orderdate",
        "t7"."o_orderpriority",
        "t7"."o_clerk",
        "t7"."o_shippriority",
        "t7"."o_comment",
        "t8"."l_orderkey",
        "t8"."l_partkey",
        "t8"."l_suppkey",
        "t8"."l_linenumber",
        "t8"."l_quantity",
        "t8"."l_extendedprice",
        "t8"."l_discount",
        "t8"."l_tax",
        "t8"."l_returnflag",
        "t8"."l_linestatus",
        "t8"."l_shipdate",
        "t8"."l_commitdate",
        "t8"."l_receiptdate",
        "t8"."l_shipinstruct",
        "t8"."l_shipmode",
        "t8"."l_comment"
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
      ) AS "t6"
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
      ) AS "t7"
        ON "t6"."c_custkey" = "t7"."o_custkey"
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
      ) AS "t8"
        ON "t8"."l_orderkey" = "t7"."o_orderkey"
    ) AS "t9"
    WHERE
      "t9"."c_mktsegment" = 'BUILDING'
      AND "t9"."o_orderdate" < FROM_ISO8601_DATE('1995-03-15')
      AND "t9"."l_shipdate" > FROM_ISO8601_DATE('1995-03-15')
  ) AS "t10"
  GROUP BY
    1,
    2,
    3
) AS "t11"
ORDER BY
  "t11"."revenue" DESC,
  "t11"."o_orderdate" ASC
LIMIT 10