SELECT
  "t0"."a",
  "t0"."b"
FROM "my_table" AS "t0"
WHERE
  "t0"."a" < CAST(100 AS TINYINT)
  AND "t0"."a" = (
    SELECT
      MAX("t1"."a") AS "Max(a)"
    FROM (
      SELECT
        "t0"."a",
        "t0"."b"
      FROM "my_table" AS "t0"
      WHERE
        "t0"."a" < CAST(100 AS TINYINT)
    ) AS "t1"
  )
  AND "t0"."b" = 'a'