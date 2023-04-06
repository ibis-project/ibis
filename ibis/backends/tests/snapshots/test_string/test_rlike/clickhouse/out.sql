SELECT
  *
FROM functional_alltypes AS t0
WHERE
  multiMatchAny(t0.string_col, [  '0'])