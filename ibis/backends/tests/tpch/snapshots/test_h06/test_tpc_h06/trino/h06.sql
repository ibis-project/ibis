SELECT
  SUM("t2"."l_extendedprice" * "t2"."l_discount") AS "revenue"
FROM (
  SELECT
    *
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
  ) AS "t1"
  WHERE
    "t1"."l_shipdate" >= FROM_ISO8601_DATE('1994-01-01')
    AND "t1"."l_shipdate" < FROM_ISO8601_DATE('1995-01-01')
    AND "t1"."l_discount" BETWEEN CAST(0.05 AS DOUBLE) AND CAST(0.07 AS DOUBLE)
    AND "t1"."l_quantity" < 24
) AS "t2"