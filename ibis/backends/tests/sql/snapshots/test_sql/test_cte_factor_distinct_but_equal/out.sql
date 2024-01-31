SELECT
  "t3"."g",
  "t3"."metric"
FROM (
  SELECT
    "t0"."g",
    SUM("t0"."f") AS "metric"
  FROM "alltypes" AS "t0"
  GROUP BY
    1
) AS "t3"
INNER JOIN (
  SELECT
    "t1"."g",
    SUM("t1"."f") AS "metric"
  FROM "alltypes" AS "t1"
  GROUP BY
    1
) AS "t6"
  ON "t3"."g" = "t6"."g"