SELECT
  "t18"."o_year",
  "t18"."mkt_share"
FROM (
  SELECT
    "t17"."o_year",
    SUM("t17"."nation_volume") / SUM("t17"."volume") AS "mkt_share"
  FROM (
    SELECT
      "t16"."o_year",
      "t16"."volume",
      "t16"."nation",
      "t16"."r_name",
      "t16"."o_orderdate",
      "t16"."p_type",
      CASE WHEN "t16"."nation" = 'BRAZIL' THEN "t16"."volume" ELSE CAST(0 AS TINYINT) END AS "nation_volume"
    FROM (
      SELECT
        EXTRACT(year FROM "t10"."o_orderdate") AS "o_year",
        "t8"."l_extendedprice" * (
          CAST(1 AS TINYINT) - "t8"."l_discount"
        ) AS "volume",
        "t15"."n_name" AS "nation",
        "t14"."r_name",
        "t10"."o_orderdate",
        "t7"."p_type"
      FROM "part" AS "t7"
      INNER JOIN "lineitem" AS "t8"
        ON "t7"."p_partkey" = "t8"."l_partkey"
      INNER JOIN "supplier" AS "t9"
        ON "t9"."s_suppkey" = "t8"."l_suppkey"
      INNER JOIN "orders" AS "t10"
        ON "t8"."l_orderkey" = "t10"."o_orderkey"
      INNER JOIN "customer" AS "t11"
        ON "t10"."o_custkey" = "t11"."c_custkey"
      INNER JOIN "nation" AS "t12"
        ON "t11"."c_nationkey" = "t12"."n_nationkey"
      INNER JOIN "region" AS "t14"
        ON "t12"."n_regionkey" = "t14"."r_regionkey"
      INNER JOIN "nation" AS "t15"
        ON "t9"."s_nationkey" = "t15"."n_nationkey"
    ) AS "t16"
    WHERE
      "t16"."r_name" = 'AMERICA'
      AND "t16"."o_orderdate" BETWEEN MAKE_DATE(1995, 1, 1) AND MAKE_DATE(1996, 12, 31)
      AND "t16"."p_type" = 'ECONOMY ANODIZED STEEL'
  ) AS "t17"
  GROUP BY
    1
) AS "t18"
ORDER BY
  "t18"."o_year" ASC