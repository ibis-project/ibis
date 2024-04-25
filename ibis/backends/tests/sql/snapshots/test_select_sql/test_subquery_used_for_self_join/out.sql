WITH "t1" AS (
  SELECT
    "t0"."g",
    "t0"."a",
    "t0"."b",
    SUM("t0"."f") AS "total"
  FROM "alltypes" AS "t0"
  GROUP BY
    1,
    2,
    3
)
SELECT
  "t5"."g",
  MAX("t5"."total" - "t5"."total_right") AS "metric"
FROM (
  SELECT
    "t3"."g",
    "t3"."a",
    "t3"."b",
    "t3"."total",
    "t4"."g" AS "g_right",
    "t4"."a" AS "a_right",
    "t4"."b" AS "b_right",
    "t4"."total" AS "total_right"
  FROM "t1" AS "t3"
  INNER JOIN "t1" AS "t4"
    ON "t3"."a" = "t4"."b"
) AS "t5"
GROUP BY
  1