WITH "t1" AS (
  SELECT
    *
  FROM "my_table" AS "t0"
  WHERE
    "t0"."a" < CAST(100 AS TINYINT)
)
SELECT
  *
FROM "t1" AS "t2"
WHERE
  "t2"."a" = (
    SELECT
      MAX("t2"."a") AS "Max(a)"
    FROM "t1" AS "t2"
  )