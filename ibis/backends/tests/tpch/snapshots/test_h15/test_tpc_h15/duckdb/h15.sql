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
        CAST(1 AS TINYINT) - "t3"."l_discount"
      )) AS "total_revenue"
    FROM (
      SELECT
        *
      FROM "lineitem" AS "t1"
      WHERE
        "t1"."l_shipdate" >= MAKE_DATE(1996, 1, 1)
        AND "t1"."l_shipdate" < MAKE_DATE(1996, 4, 1)
    ) AS "t3"
    GROUP BY
      1
  ) AS "t5"
    ON "t2"."s_suppkey" = "t5"."l_suppkey"
)
SELECT
  *
FROM (
  SELECT
    "t9"."s_suppkey",
    "t9"."s_name",
    "t9"."s_address",
    "t9"."s_phone",
    "t9"."total_revenue"
  FROM (
    SELECT
      *
    FROM "t6" AS "t7"
    WHERE
      "t7"."total_revenue" = (
        SELECT
          MAX("t7"."total_revenue") AS "Max(total_revenue)"
        FROM "t6" AS "t7"
      )
  ) AS "t9"
) AS "t10"
ORDER BY
  "t10"."s_suppkey" ASC