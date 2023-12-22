SELECT
  CAST(t1.string_col AS Nullable(Float64)) AS "Cast(string_col, float64)"
FROM (
  SELECT
    t0.string_col,
    COUNT(*) AS count
  FROM functional_alltypes AS t0
  GROUP BY
    t0.string_col
) AS t1