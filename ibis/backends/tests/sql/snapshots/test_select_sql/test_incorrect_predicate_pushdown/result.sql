SELECT
  "t1"."x"
FROM (
  SELECT
    "t0"."x" + CAST(1 AS TINYINT) AS "x"
  FROM "t" AS "t0"
) AS "t1"
WHERE
  "t1"."x" > CAST(1 AS TINYINT)