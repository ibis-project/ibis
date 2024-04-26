WITH "t1" AS (
  SELECT
    "t0"."a",
    "t0"."b"
  FROM "my_table" AS "t0"
  WHERE
    "t0"."a" < CAST(100 AS TINYINT)
)
SELECT
  "t2"."a",
  "t2"."b"
FROM "t1" AS "t2"
WHERE
  "t2"."a" = (
    SELECT
      MAX("t2"."a") AS "Max(a)"
    FROM "t1" AS "t2"
  )