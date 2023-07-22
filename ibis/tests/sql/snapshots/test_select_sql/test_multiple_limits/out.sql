SELECT t0.*
FROM (
  SELECT t1.*
  FROM functional_alltypes t1
  LIMIT 20
) t0
LIMIT 10