SELECT DISTINCT
  *
FROM (
  SELECT
    t0.string_col AS string_col,
    t0.int_col AS int_col
  FROM functional_alltypes AS t0
) AS t1