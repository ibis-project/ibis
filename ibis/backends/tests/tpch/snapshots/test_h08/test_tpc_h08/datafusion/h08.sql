SELECT
  *
FROM (
  SELECT
    "t16"."o_year",
    CAST(SUM("t16"."nation_volume") AS DOUBLE PRECISION) / SUM("t16"."volume") AS "mkt_share"
  FROM (
    SELECT
      "t16"."volume",
      "t16"."nation",
      "t16"."r_name",
      "t16"."o_orderdate",
      "t16"."p_type",
      "t16"."nation_volume",
      "t16"."o_year"
    FROM (
      SELECT
        "t15"."o_year",
        "t15"."volume",
        "t15"."nation",
        "t15"."r_name",
        "t15"."o_orderdate",
        "t15"."p_type",
        CASE WHEN "t15"."nation" = 'BRAZIL' THEN "t15"."volume" ELSE 0 END AS "nation_volume"
      FROM (
        SELECT
          DATE_PART('year', "t10"."o_orderdate") AS "o_year",
          "t8"."l_extendedprice" * (
            1 - "t8"."l_discount"
          ) AS "volume",
          "t13"."n_name" AS "nation",
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
        INNER JOIN "nation" AS "t13"
          ON "t9"."s_nationkey" = "t13"."n_nationkey"
      ) AS "t15"
      WHERE
        "t15"."r_name" = 'AMERICA'
        AND "t15"."o_orderdate" BETWEEN DATE_TRUNC('DAY', '1995-01-01') AND DATE_TRUNC('DAY', '1996-12-31')
        AND "t15"."p_type" = 'ECONOMY ANODIZED STEEL'
    ) AS "t16"
  ) AS t16
  GROUP BY
    "t16"."o_year"
) AS "t17"
ORDER BY
  "t17"."o_year" ASC