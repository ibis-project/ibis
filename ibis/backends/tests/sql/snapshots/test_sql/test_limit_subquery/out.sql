SELECT
  *
FROM (
  SELECT
    *
  FROM "star1" AS "t0"
  LIMIT 10
) AS "t1"
WHERE
  "t1"."f" > 0