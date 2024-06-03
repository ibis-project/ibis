SELECT
  CAST((
    SUM(
      CASE
        WHEN "t5"."p_type" LIKE 'PROMO%'
        THEN "t5"."l_extendedprice" * (
          1 - "t5"."l_discount"
        )
        ELSE 0
      END
    ) * 100
  ) AS DOUBLE PRECISION) / SUM("t5"."l_extendedprice" * (
    1 - "t5"."l_discount"
  )) AS "promo_revenue"
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t2"."l_orderkey",
      "t2"."l_partkey",
      "t2"."l_suppkey",
      "t2"."l_linenumber",
      "t2"."l_quantity",
      "t2"."l_extendedprice",
      "t2"."l_discount",
      "t2"."l_tax",
      "t2"."l_returnflag",
      "t2"."l_linestatus",
      "t2"."l_shipdate",
      "t2"."l_commitdate",
      "t2"."l_receiptdate",
      "t2"."l_shipinstruct",
      "t2"."l_shipmode",
      "t2"."l_comment",
      "t3"."p_partkey",
      "t3"."p_name",
      "t3"."p_mfgr",
      "t3"."p_brand",
      "t3"."p_type",
      "t3"."p_size",
      "t3"."p_container",
      "t3"."p_retailprice",
      "t3"."p_comment"
    FROM "lineitem" AS "t2"
    INNER JOIN "part" AS "t3"
      ON "t2"."l_partkey" = "t3"."p_partkey"
  ) AS "t4"
  WHERE
    "t4"."l_shipdate" >= DATE_TRUNC('DAY', '1995-09-01')
    AND "t4"."l_shipdate" < DATE_TRUNC('DAY', '1995-10-01')
) AS "t5"