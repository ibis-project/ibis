SELECT
  "t3"."n_nationkey",
  "t3"."n_name",
  "t3"."n_regionkey",
  "t3"."n_comment"
FROM "tpch_region" AS "t2"
FULL OUTER JOIN "tpch_nation" AS "t3"
  ON "t2"."r_regionkey" = "t3"."n_regionkey"