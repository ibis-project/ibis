WITH "t6" AS (
  SELECT
    "t2"."s_suppkey",
    "t2"."s_name",
    "t2"."s_address",
    "t2"."s_nationkey",
    "t2"."s_phone",
    "t2"."s_acctbal",
    "t2"."s_comment",
    "t5"."l_suppkey",
    "t5"."total_revenue"
  FROM "supplier" AS "t2"
  INNER JOIN (
    SELECT
      "t3"."l_suppkey",
      SUM("t3"."l_extendedprice" * (
        1 - "t3"."l_discount"
      )) AS "total_revenue"
    FROM (
      SELECT
        "t3"."l_orderkey",
        "t3"."l_partkey",
        "t3"."l_linenumber",
        "t3"."l_quantity",
        "t3"."l_extendedprice",
        "t3"."l_discount",
        "t3"."l_tax",
        "t3"."l_returnflag",
        "t3"."l_linestatus",
        "t3"."l_shipdate",
        "t3"."l_commitdate",
        "t3"."l_receiptdate",
        "t3"."l_shipinstruct",
        "t3"."l_shipmode",
        "t3"."l_comment",
        "t3"."l_suppkey"
      FROM (
        SELECT
          *
        FROM "lineitem" AS "t1"
        WHERE
          "t1"."l_shipdate" >= DATE_TRUNC('DAY', '1996-01-01')
          AND "t1"."l_shipdate" < DATE_TRUNC('DAY', '1996-04-01')
      ) AS "t3"
    ) AS t3
    GROUP BY
      "t3"."l_suppkey"
  ) AS "t5"
    ON "t2"."s_suppkey" = "t5"."l_suppkey"
)
SELECT
  "t7"."s_suppkey",
  "t7"."s_name",
  "t7"."s_address",
  "t7"."s_phone",
  "t7"."total_revenue"
FROM "t6" AS "t7"
WHERE
  "t7"."total_revenue" = (
    SELECT
      MAX("t7"."total_revenue") AS "Max(total_revenue)"
    FROM "t6" AS "t7"
  )
ORDER BY
  "t7"."s_suppkey" ASC