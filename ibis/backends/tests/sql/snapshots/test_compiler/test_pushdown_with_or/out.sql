SELECT
  "t1"."double_col",
  "t1"."string_col",
  "t1"."int_col",
  "t1"."float_col"
FROM (
  SELECT
    "t0"."double_col",
    "t0"."string_col",
    "t0"."int_col",
    "t0"."float_col"
  FROM "functional_alltypes" AS "t0"
  WHERE
    "t0"."double_col" > CAST(3.14 AS DOUBLE) AND CONTAINS("t0"."string_col", 'foo')
) AS "t1"
WHERE
  (
    (
      "t1"."int_col" - CAST(1 AS TINYINT)
    ) = CAST(0 AS TINYINT)
  )
  OR (
    "t1"."float_col" <= CAST(1.34 AS DOUBLE)
  )