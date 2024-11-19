SELECT
  CAST(ROUND("t0"."double_col", 0) AS Nullable(Int64)) AS "Round(double_col, 0)"
FROM "functional_alltypes" AS "t0"