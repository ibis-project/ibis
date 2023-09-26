SELECT
  SUM(t0.float_col) AS "Sum(float_col)"
FROM functional_alltypes AS t0
WHERE
  t0.int_col > 0