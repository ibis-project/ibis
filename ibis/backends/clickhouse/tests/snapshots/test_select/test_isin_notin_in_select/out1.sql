SELECT
  t0.id AS id,
  t0.bool_col AS bool_col,
  t0.tinyint_col AS tinyint_col,
  t0.smallint_col AS smallint_col,
  t0.int_col AS int_col,
  t0.bigint_col AS bigint_col,
  t0.float_col AS float_col,
  t0.double_col AS double_col,
  t0.date_string_col AS date_string_col,
  t0.string_col AS string_col,
  t0.timestamp_col AS timestamp_col,
  t0.year AS year,
  t0.month AS month
FROM functional_alltypes AS t0
WHERE
  t0.string_col IN ('foo', 'bar')