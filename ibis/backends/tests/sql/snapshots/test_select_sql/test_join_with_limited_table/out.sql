SELECT
  "t4"."c",
  "t4"."f",
  "t4"."foo_id",
  "t4"."bar_id"
FROM (
  SELECT
    *
  FROM "star1" AS "t0"
  LIMIT 100
) AS "t4"
INNER JOIN "star2" AS "t3"
  ON "t4"."foo_id" = "t3"."foo_id"