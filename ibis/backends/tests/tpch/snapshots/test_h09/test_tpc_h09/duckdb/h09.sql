SELECT
  "t14"."nation",
  "t14"."o_year",
  "t14"."sum_profit"
FROM (
  SELECT
    "t13"."nation",
    "t13"."o_year",
    SUM("t13"."amount") AS "sum_profit"
  FROM (
    SELECT
      "t12"."amount",
      "t12"."o_year",
      "t12"."nation",
      "t12"."p_name"
    FROM (
      SELECT
        (
          "t6"."l_extendedprice" * (
            CAST(1 AS TINYINT) - "t6"."l_discount"
          )
        ) - (
          "t8"."ps_supplycost" * "t6"."l_quantity"
        ) AS "amount",
        EXTRACT(year FROM "t10"."o_orderdate") AS "o_year",
        "t11"."n_name" AS "nation",
        "t9"."p_name"
      FROM "lineitem" AS "t6"
      INNER JOIN "supplier" AS "t7"
        ON "t7"."s_suppkey" = "t6"."l_suppkey"
      INNER JOIN "partsupp" AS "t8"
        ON "t8"."ps_suppkey" = "t6"."l_suppkey" AND "t8"."ps_partkey" = "t6"."l_partkey"
      INNER JOIN "part" AS "t9"
        ON "t9"."p_partkey" = "t6"."l_partkey"
      INNER JOIN "orders" AS "t10"
        ON "t10"."o_orderkey" = "t6"."l_orderkey"
      INNER JOIN "nation" AS "t11"
        ON "t7"."s_nationkey" = "t11"."n_nationkey"
    ) AS "t12"
    WHERE
      "t12"."p_name" LIKE '%green%'
  ) AS "t13"
  GROUP BY
    1,
    2
) AS "t14"
ORDER BY
  "t14"."nation" ASC,
  "t14"."o_year" DESC