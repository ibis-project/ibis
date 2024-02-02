SELECT
  "t1"."c",
  "t1"."f",
  "t1"."foo_id",
  "t1"."bar_id"
FROM (
  SELECT
    *
  FROM "star1" AS "t0"
  LIMIT 10
) AS "t1"
WHERE
  "t1"."f" > CAST(0 AS TINYINT)