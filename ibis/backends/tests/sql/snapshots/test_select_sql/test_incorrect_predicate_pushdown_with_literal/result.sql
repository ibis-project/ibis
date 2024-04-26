SELECT
  *
FROM (
  SELECT
    CAST(1 AS TINYINT) AS "a"
  FROM "t" AS "t0"
) AS "t1"
WHERE
  "t1"."a" > "t1"."a"