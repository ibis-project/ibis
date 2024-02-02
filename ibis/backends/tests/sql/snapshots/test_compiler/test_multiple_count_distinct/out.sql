SELECT
  "t0"."string_col",
  COUNT(DISTINCT "t0"."int_col") AS "int_card",
  COUNT(DISTINCT "t0"."smallint_col") AS "smallint_card"
FROM "functional_alltypes" AS "t0"
GROUP BY
  1