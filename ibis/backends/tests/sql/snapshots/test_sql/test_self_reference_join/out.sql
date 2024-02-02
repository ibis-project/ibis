SELECT
  "t1"."c",
  "t1"."f",
  "t1"."foo_id",
  "t1"."bar_id"
FROM "star1" AS "t1"
INNER JOIN "star1" AS "t3"
  ON "t1"."foo_id" = "t3"."bar_id"