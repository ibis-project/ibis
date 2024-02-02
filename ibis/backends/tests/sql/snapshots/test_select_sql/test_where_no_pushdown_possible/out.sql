SELECT
  "t4"."c",
  "t4"."f",
  "t4"."foo_id",
  "t4"."bar_id",
  "t4"."diff"
FROM (
  SELECT
    "t2"."c",
    "t2"."f",
    "t2"."foo_id",
    "t2"."bar_id",
    "t2"."f" - "t3"."value1" AS "diff"
  FROM "star1" AS "t2"
  INNER JOIN "star2" AS "t3"
    ON "t2"."foo_id" = "t3"."foo_id"
) AS "t4"
WHERE
  "t4"."diff" > CAST(1 AS TINYINT)