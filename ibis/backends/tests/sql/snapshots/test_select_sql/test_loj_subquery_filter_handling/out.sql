SELECT
  "t4"."id" AS "left_id",
  "t4"."desc" AS "left_desc",
  "t5"."id" AS "right_id",
  "t5"."desc" AS "right_desc"
FROM (
  SELECT
    "t0"."id",
    "t0"."desc"
  FROM "foo" AS "t0"
  WHERE
    "t0"."id" < CAST(2 AS TINYINT)
) AS "t4"
LEFT OUTER JOIN (
  SELECT
    "t1"."id",
    "t1"."desc"
  FROM "bar" AS "t1"
  WHERE
    "t1"."id" < CAST(3 AS TINYINT)
) AS "t5"
  ON "t4"."id" = "t5"."id" AND "t4"."desc" = "t5"."desc"