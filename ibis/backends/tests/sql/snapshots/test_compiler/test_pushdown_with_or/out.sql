SELECT
  *
FROM "functional_alltypes" AS "t0"
WHERE
  "t0"."double_col" > 3.14
  AND CONTAINS("t0"."string_col", 'foo')
  AND (
    (
      (
        "t0"."int_col" - 1
      ) = 0
    ) OR (
      "t0"."float_col" <= 1.34
    )
  )