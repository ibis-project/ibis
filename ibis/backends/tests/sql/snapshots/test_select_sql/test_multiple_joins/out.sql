SELECT
  "t3"."c",
  "t3"."f",
  "t3"."foo_id",
  "t3"."bar_id",
  "t4"."value1",
  "t5"."value2"
FROM "star1" AS "t3"
LEFT OUTER JOIN "star2" AS "t4"
  ON "t3"."foo_id" = "t4"."foo_id"
INNER JOIN "star3" AS "t5"
  ON "t3"."bar_id" = "t5"."bar_id"