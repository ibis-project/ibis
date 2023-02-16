SELECT
  t0.int_col,
  t0.bigint_col
FROM (
  SELECT
    t1.int_col AS int_col,
    SUM(t1.bigint_col) AS bigint_col
  FROM t AS t1
  GROUP BY
    1
) AS t0
WHERE
  t0.bigint_col = 60