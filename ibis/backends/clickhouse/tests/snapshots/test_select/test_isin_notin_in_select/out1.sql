SELECT
  *
FROM ibis_testing.functional_alltypes AS t0
WHERE
  t0.string_col IN ('foo', 'bar')