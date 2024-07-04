SELECT
  CASE
    WHEN "t0"."float_col" > 0
    THEN "t0"."int_col" * 2
    WHEN "t0"."float_col" < 0
    THEN "t0"."int_col"
    ELSE 0
  END AS "SearchedCase((Greater(float_col, 0), Less(float_col, 0)), (Multiply(int_col, 2), int_col), 0)"
FROM "functional_alltypes" AS "t0"