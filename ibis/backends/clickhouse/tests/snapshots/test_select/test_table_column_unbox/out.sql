SELECT
  t1.string_col
FROM (
  SELECT
    t0.string_col,
    SUM(t0.float_col) AS total
  FROM functional_alltypes AS t0
  WHERE
    t0.int_col > 0
  GROUP BY
    t0.string_col
) AS t1