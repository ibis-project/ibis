SELECT t0.*
FROM functional_alltypes t0
WHERE EXISTS (
  SELECT 1
  FROM functional_alltypes t1
  WHERE t0.`string_col` = t1.`string_col`
)
