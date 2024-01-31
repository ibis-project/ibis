SELECT
  CAST(CEIL("t0"."double_col") AS Nullable(Int64)) AS "Ceil(double_col)"
FROM "functional_alltypes" AS "t0"