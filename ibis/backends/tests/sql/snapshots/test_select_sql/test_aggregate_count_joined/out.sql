SELECT
  COUNT(*) AS "CountStar()"
FROM (
  SELECT
    "t3"."n_nationkey",
    "t3"."n_name",
    "t3"."n_regionkey",
    "t3"."n_comment",
    "t2"."r_name" AS "region"
  FROM "tpch_region" AS "t2"
  INNER JOIN "tpch_nation" AS "t3"
    ON "t2"."r_regionkey" = "t3"."n_regionkey"
) AS "t4"