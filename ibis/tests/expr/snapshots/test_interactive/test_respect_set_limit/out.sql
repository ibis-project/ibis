SELECT t0.*
FROM (
  SELECT t1.`id`, t1.`bool_col`
  FROM functional_alltypes t1
  LIMIT 10
) t0
LIMIT 11