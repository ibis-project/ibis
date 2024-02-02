SELECT
  "t0"."double_col",
  "t0"."string_col",
  "t0"."int_col",
  "t0"."float_col"
FROM "functional_alltypes" AS "t0"
WHERE
  "t0"."double_col" > CAST(3.14 AS DOUBLE)
  AND CONTAINS("t0"."string_col", 'foo')
  AND (
    (
      (
        "t0"."int_col" - CAST(1 AS TINYINT)
      ) = CAST(0 AS TINYINT)
    )
    OR (
      "t0"."float_col" <= CAST(1.34 AS DOUBLE)
    )
  )