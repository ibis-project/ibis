SELECT
  "t4"."c",
  "t4"."f",
  "t4"."foo_id",
  "t4"."bar_id",
  "t4"."value1",
  "t4"."value3"
FROM (
  SELECT
    "t2"."c",
    "t2"."f",
    "t2"."foo_id",
    "t2"."bar_id",
    "t3"."value1",
    "t3"."value3"
  FROM "star1" AS "t2"
  INNER JOIN "star2" AS "t3"
    ON "t2"."foo_id" = "t3"."foo_id"
) AS "t4"
WHERE
  "t4"."f" > CAST(0 AS TINYINT) AND "t4"."value3" < CAST(1000 AS SMALLINT)