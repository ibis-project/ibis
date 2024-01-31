SELECT
  SUM("t1"."l_extendedprice" * "t1"."l_discount") AS "revenue"
FROM (
  SELECT
    "t0"."l_orderkey",
    "t0"."l_partkey",
    "t0"."l_suppkey",
    "t0"."l_linenumber",
    "t0"."l_quantity",
    "t0"."l_extendedprice",
    "t0"."l_discount",
    "t0"."l_tax",
    "t0"."l_returnflag",
    "t0"."l_linestatus",
    "t0"."l_shipdate",
    "t0"."l_commitdate",
    "t0"."l_receiptdate",
    "t0"."l_shipinstruct",
    "t0"."l_shipmode",
    "t0"."l_comment"
  FROM "lineitem" AS "t0"
  WHERE
    "t0"."l_shipdate" >= MAKE_DATE(1994, 1, 1)
    AND "t0"."l_shipdate" < MAKE_DATE(1995, 1, 1)
    AND "t0"."l_discount" BETWEEN CAST(0.05 AS DOUBLE) AND CAST(0.07 AS DOUBLE)
    AND "t0"."l_quantity" < CAST(24 AS TINYINT)
) AS "t1"