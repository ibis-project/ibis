SELECT
  *
FROM functional_alltypes AS t0
WHERE
  t0.float_col > 0 AND t0.int_col < (
    t0.float_col * 2
  )