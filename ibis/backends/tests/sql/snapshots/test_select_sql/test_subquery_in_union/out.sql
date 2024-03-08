WITH "t1" AS (
  SELECT
    "t0"."a",
    "t0"."g",
    SUM("t0"."f") AS "metric"
  FROM "alltypes" AS "t0"
  GROUP BY
    1,
    2
), "t6" AS (
  SELECT
    "t3"."a",
    "t3"."g",
    "t3"."metric"
  FROM "t1" AS "t3"
  INNER JOIN "t1" AS "t5"
    ON "t3"."g" = "t5"."g"
)
SELECT
  "t9"."a",
  "t9"."g",
  "t9"."metric"
FROM (
  SELECT
    *
  FROM "t6" AS "t7"
  UNION ALL
  SELECT
    *
  FROM "t6" AS "t8"
) AS "t9"