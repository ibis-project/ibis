SELECT
  CAST(ROUND("t0"."double_col", 2) AS Nullable(Float64)) AS "Round(double_col, 2)"
FROM "functional_alltypes" AS "t0"