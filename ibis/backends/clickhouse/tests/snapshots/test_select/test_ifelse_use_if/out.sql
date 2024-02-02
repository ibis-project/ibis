SELECT
  CASE WHEN "t0"."float_col" > 0 THEN "t0"."int_col" ELSE "t0"."bigint_col" END AS "IfElse(Greater(float_col, 0), int_col, bigint_col)"
FROM "functional_alltypes" AS "t0"