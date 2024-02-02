WITH "t1" AS (
  SELECT
    "t0"."a",
    "t0"."b"
  FROM "t" AS "t0"
  ORDER BY
    "t0"."b" ASC
)
SELECT
  "t3"."a",
  "t3"."b"
FROM (
  SELECT
    *
  FROM "t1" AS "t2"
  UNION ALL
  SELECT
    *
  FROM "t1" AS "t2"
) AS "t3"