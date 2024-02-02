SELECT
  "t3"."c_name",
  "t5"."r_name",
  "t4"."n_name"
FROM "tpch_customer" AS "t3"
INNER JOIN "tpch_nation" AS "t4"
  ON "t3"."c_nationkey" = "t4"."n_nationkey"
INNER JOIN "tpch_region" AS "t5"
  ON "t4"."n_regionkey" = "t5"."r_regionkey"
SEMI JOIN (
  SELECT
    "t7"."n_name",
    "t7"."Sum(Cast(c_acctbal, float64))"
  FROM (
    SELECT
      "t6"."n_name",
      SUM(CAST("t6"."c_acctbal" AS DOUBLE)) AS "Sum(Cast(c_acctbal, float64))"
    FROM (
      SELECT
        "t3"."c_custkey",
        "t3"."c_name",
        "t3"."c_address",
        "t3"."c_nationkey",
        "t3"."c_phone",
        "t3"."c_acctbal",
        "t3"."c_mktsegment",
        "t3"."c_comment",
        "t4"."n_name",
        "t5"."r_name"
      FROM "tpch_customer" AS "t3"
      INNER JOIN "tpch_nation" AS "t4"
        ON "t3"."c_nationkey" = "t4"."n_nationkey"
      INNER JOIN "tpch_region" AS "t5"
        ON "t4"."n_regionkey" = "t5"."r_regionkey"
    ) AS "t6"
    GROUP BY
      1
  ) AS "t7"
  ORDER BY
    "t7"."Sum(Cast(c_acctbal, float64))" DESC
  LIMIT 10
) AS "t10"
  ON "t4"."n_name" = "t10"."n_name"