WITH "t1" AS (
  SELECT
    *
  FROM "t" AS "t0"
  ORDER BY
    "t0"."b" ASC
)
SELECT
  *
FROM "t1" AS "t2"
UNION ALL
SELECT
  *
FROM "t1" AS "t2"