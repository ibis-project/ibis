WITH "t1" AS (
  SELECT
    "t0"."a",
    "t0"."g",
    SUM("t0"."f") AS "metric"
  FROM "alltypes" AS "t0"
  GROUP BY
    1,
    2
), "t5" AS (
  SELECT
    "t3"."a",
    "t3"."g",
    "t3"."metric"
  FROM "t1" AS "t3"
  INNER JOIN "t1" AS "t4"
    ON "t3"."g" = "t4"."g"
)
SELECT
  *
FROM "t5" AS "t6"
UNION ALL
SELECT
  *
FROM "t5" AS "t7"