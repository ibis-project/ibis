SELECT
  "t1"."string_col",
  COUNT(DISTINCT "t1"."int_col") AS "nunique"
FROM (
  SELECT
    *
  FROM "functional_alltypes" AS "t0"
  WHERE
    "t0"."bigint_col" > 0
) AS "t1"
GROUP BY
  1