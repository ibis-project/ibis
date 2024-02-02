WITH "t9" AS (
  SELECT
    "t8"."region",
    EXTRACT(year FROM "t8"."odate") AS "year",
    CAST(SUM("t8"."amount") AS DOUBLE) AS "total"
  FROM (
    SELECT
      "t4"."r_name" AS "region",
      "t5"."n_name" AS "nation",
      "t7"."o_totalprice" AS "amount",
      CAST("t7"."o_orderdate" AS TIMESTAMP) AS "odate"
    FROM "tpch_region" AS "t4"
    INNER JOIN "tpch_nation" AS "t5"
      ON "t4"."r_regionkey" = "t5"."n_regionkey"
    INNER JOIN "tpch_customer" AS "t6"
      ON "t6"."c_nationkey" = "t5"."n_nationkey"
    INNER JOIN "tpch_orders" AS "t7"
      ON "t7"."o_custkey" = "t6"."c_custkey"
  ) AS "t8"
  GROUP BY
    1,
    2
)
SELECT
  "t11"."region",
  "t11"."year",
  "t11"."total" - "t13"."total" AS "yoy_change"
FROM "t9" AS "t11"
INNER JOIN "t9" AS "t13"
  ON "t11"."year" = (
    "t13"."year" - CAST(1 AS TINYINT)
  )