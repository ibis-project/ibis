SELECT
  CAST(t1.string_col AS Nullable(Float64))
FROM (
  SELECT
    t0.string_col,
    COUNT(*) AS count
  FROM ibis_testing.functional_alltypes AS t0
  GROUP BY
    1
) AS t1