SELECT
  "t14"."supp_nation",
  "t14"."cust_nation",
  "t14"."l_year",
  "t14"."revenue"
FROM (
  SELECT
    "t13"."supp_nation",
    "t13"."cust_nation",
    "t13"."l_year",
    SUM("t13"."volume") AS "revenue"
  FROM (
    SELECT
      "t12"."supp_nation",
      "t12"."cust_nation",
      "t12"."l_shipdate",
      "t12"."l_extendedprice",
      "t12"."l_discount",
      "t12"."l_year",
      "t12"."volume"
    FROM (
      SELECT
        "t9"."n_name" AS "supp_nation",
        "t11"."n_name" AS "cust_nation",
        "t6"."l_shipdate",
        "t6"."l_extendedprice",
        "t6"."l_discount",
        EXTRACT(year FROM "t6"."l_shipdate") AS "l_year",
        "t6"."l_extendedprice" * (
          CAST(1 AS TINYINT) - "t6"."l_discount"
        ) AS "volume"
      FROM "supplier" AS "t5"
      INNER JOIN "lineitem" AS "t6"
        ON "t5"."s_suppkey" = "t6"."l_suppkey"
      INNER JOIN "orders" AS "t7"
        ON "t7"."o_orderkey" = "t6"."l_orderkey"
      INNER JOIN "customer" AS "t8"
        ON "t8"."c_custkey" = "t7"."o_custkey"
      INNER JOIN "nation" AS "t9"
        ON "t5"."s_nationkey" = "t9"."n_nationkey"
      INNER JOIN "nation" AS "t11"
        ON "t8"."c_nationkey" = "t11"."n_nationkey"
    ) AS "t12"
    WHERE
      (
        (
          (
            "t12"."cust_nation" = 'FRANCE'
          ) AND (
            "t12"."supp_nation" = 'GERMANY'
          )
        )
        OR (
          (
            "t12"."cust_nation" = 'GERMANY'
          ) AND (
            "t12"."supp_nation" = 'FRANCE'
          )
        )
      )
      AND "t12"."l_shipdate" BETWEEN MAKE_DATE(1995, 1, 1) AND MAKE_DATE(1996, 12, 31)
  ) AS "t13"
  GROUP BY
    1,
    2,
    3
) AS "t14"
ORDER BY
  "t14"."supp_nation" ASC,
  "t14"."cust_nation" ASC,
  "t14"."l_year" ASC