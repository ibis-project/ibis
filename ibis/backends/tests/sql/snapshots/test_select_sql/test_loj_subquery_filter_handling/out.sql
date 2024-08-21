SELECT
  "t4"."id" AS "left_id",
  "t4"."desc" AS "left_desc",
  "t5"."id" AS "right_id",
  "t5"."desc" AS "right_desc"
FROM (
  SELECT
    *
  FROM "foo" AS "t0"
  WHERE
    "t0"."id" < 2
) AS "t4"
LEFT OUTER JOIN (
  SELECT
    *
  FROM "bar" AS "t1"
  WHERE
    "t1"."id" < 3
) AS "t5"
  ON "t4"."id" = "t5"."id" AND "t4"."desc" = "t5"."desc"