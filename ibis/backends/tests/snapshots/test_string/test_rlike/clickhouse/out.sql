SELECT
  *
FROM functional_alltypes AS t0
WHERE
  match(t0.string_col, '0')