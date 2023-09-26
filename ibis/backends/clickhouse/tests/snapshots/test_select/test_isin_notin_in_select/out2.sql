SELECT
  *
FROM functional_alltypes AS t0
WHERE
  NOT (
    t0.string_col IN ('foo', 'bar')
  )