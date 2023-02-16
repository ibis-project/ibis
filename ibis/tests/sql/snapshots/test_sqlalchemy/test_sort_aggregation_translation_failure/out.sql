SELECT
  t0.string_col,
  t0.foo
FROM (
  SELECT
    t1.string_col AS string_col,
    MAX(t1.double_col) AS foo
  FROM functional_alltypes AS t1
  GROUP BY
    1
) AS t0
ORDER BY
  t0.foo DESC