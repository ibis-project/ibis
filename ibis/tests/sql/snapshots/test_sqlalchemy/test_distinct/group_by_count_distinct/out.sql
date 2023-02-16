SELECT
  t0.string_col,
  COUNT(DISTINCT t0.int_col) AS nunique
FROM functional_alltypes AS t0
GROUP BY
  1