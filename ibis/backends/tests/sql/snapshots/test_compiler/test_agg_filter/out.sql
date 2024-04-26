WITH "t3" AS (
  SELECT
    "t2"."a",
    "t2"."b2"
  FROM (
    SELECT
      "t1"."a",
      "t1"."b2"
    FROM (
      SELECT
        "t0"."a",
        "t0"."b",
        "t0"."b" * CAST(2 AS TINYINT) AS "b2"
      FROM "my_table" AS "t0"
    ) AS "t1"
  ) AS "t2"
  WHERE
    "t2"."a" < CAST(100 AS TINYINT)
)
SELECT
  "t4"."a",
  "t4"."b2"
FROM "t3" AS "t4"
WHERE
  "t4"."a" = (
    SELECT
      MAX("t4"."a") AS "Max(a)"
    FROM "t3" AS "t4"
  )