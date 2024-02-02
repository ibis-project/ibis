SELECT
  "t4"."foo_id",
  SUM("t4"."value1") AS "total"
FROM (
  SELECT
    "t2"."c",
    "t2"."f",
    "t2"."foo_id",
    "t2"."bar_id",
    "t3"."value1"
  FROM "star1" AS "t2"
  INNER JOIN "star2" AS "t3"
    ON "t2"."foo_id" = "t3"."foo_id"
) AS "t4"
GROUP BY
  1