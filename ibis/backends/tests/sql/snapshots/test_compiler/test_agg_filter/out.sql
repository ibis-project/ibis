WITH "t1" AS (
  SELECT
    "t0"."a",
    "t0"."b",
    "t0"."b" * CAST(2 AS TINYINT) AS "b2"
  FROM "my_table" AS "t0"
)
SELECT
  "t2"."a",
  "t2"."b2"
FROM "t1" AS "t2"
WHERE
  "t2"."a" < CAST(100 AS TINYINT)
  AND "t2"."a" = (
    SELECT
      MAX("t3"."a") AS "Max(a)"
    FROM (
      SELECT
        "t2"."a",
        "t2"."b2"
      FROM "t1" AS "t2"
      WHERE
        "t2"."a" < CAST(100 AS TINYINT)
    ) AS "t3"
  )