SELECT
  SUM("t1"."l_extendedprice" * "t1"."l_discount") AS "revenue"
FROM (
  SELECT
    *
  FROM "lineitem" AS "t0"
  WHERE
    "t0"."l_shipdate" >= DATE_TRUNC('DAY', '1994-01-01')
    AND "t0"."l_shipdate" < DATE_TRUNC('DAY', '1995-01-01')
    AND "t0"."l_discount" BETWEEN 0.05 AND 0.07
    AND "t0"."l_quantity" < 24
) AS "t1"