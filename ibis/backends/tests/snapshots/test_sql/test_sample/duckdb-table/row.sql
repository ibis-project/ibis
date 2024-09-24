SELECT
  *
FROM (
  SELECT
    *
  FROM "test" AS "t0"
  WHERE
    "t0"."x" > 10
) AS "t1" TABLESAMPLE bernoulli (50.0 PERCENT)