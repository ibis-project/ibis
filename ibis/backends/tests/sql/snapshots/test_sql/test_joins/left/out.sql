SELECT
  "t2"."r_regionkey",
  "t2"."r_name",
  "t2"."r_comment",
  "t3"."n_nationkey",
  "t3"."n_name",
  "t3"."n_regionkey",
  "t3"."n_comment"
FROM "tpch_region" AS "t2"
LEFT OUTER JOIN "tpch_nation" AS "t3"
  ON "t2"."r_regionkey" = "t3"."n_regionkey"