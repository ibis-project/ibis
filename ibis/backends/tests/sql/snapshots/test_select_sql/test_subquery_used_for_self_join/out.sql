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
  "t6"."g",
  MAX("t6"."total" - "t6"."total_right") AS "metric"
FROM (
  SELECT
    "t3"."g",
    "t3"."a",
    "t3"."b",
    "t3"."total",
    "t5"."g" AS "g_right",
    "t5"."a" AS "a_right",
    "t5"."b" AS "b_right",
    "t5"."total" AS "total_right"
  FROM "t1" AS "t3"
  INNER JOIN "t1" AS "t5"
    ON "t3"."a" = "t5"."b"
) AS "t6"
GROUP BY
  1