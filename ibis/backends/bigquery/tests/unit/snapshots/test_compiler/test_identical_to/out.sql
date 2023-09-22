SELECT
  t0.*
FROM functional_alltypes AS t0
WHERE
  (
    t0.`string_col` IS NOT DISTINCT FROM 'a'
  )
  AND (
    t0.`date_string_col` IS NOT DISTINCT FROM 'b'
  )