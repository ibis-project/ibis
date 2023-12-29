SELECT
  "t13"."l_orderkey",
  "t13"."revenue",
  "t13"."o_orderdate",
  "t13"."o_shippriority"
FROM (
  SELECT
    "t12"."l_orderkey",
    "t12"."o_orderdate",
    "t12"."o_shippriority",
    SUM("t12"."l_extendedprice" * (
      1 - "t12"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t11"."c_custkey",
      "t11"."c_name",
      "t11"."c_address",
      "t11"."c_nationkey",
      "t11"."c_phone",
      "t11"."c_acctbal",
      "t11"."c_mktsegment",
      "t11"."c_comment",
      "t11"."o_orderkey",
      "t11"."o_custkey",
      "t11"."o_orderstatus",
      "t11"."o_totalprice",
      "t11"."o_orderdate",
      "t11"."o_orderpriority",
      "t11"."o_clerk",
      "t11"."o_shippriority",
      "t11"."o_comment",
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
      "t11"."l_comment"
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
        FROM "customer" AS "t0"
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
        FROM "orders" AS "t1"
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
        FROM "lineitem" AS "t2"
      ) AS "t8"
        ON "t8"."l_orderkey" = "t7"."o_orderkey"
    ) AS "t11"
    WHERE
      "t11"."c_mktsegment" = 'BUILDING'
      AND "t11"."o_orderdate" < FROM_ISO8601_DATE('1995-03-15')
      AND "t11"."l_shipdate" > FROM_ISO8601_DATE('1995-03-15')
  ) AS "t12"
  GROUP BY
    1,
    2,
    3
) AS "t13"
ORDER BY
  "t13"."revenue" DESC,
  "t13"."o_orderdate" ASC
LIMIT 10