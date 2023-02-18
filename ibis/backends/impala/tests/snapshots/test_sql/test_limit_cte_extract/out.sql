SELECT t0.*
FROM (
  SELECT t2.*
  FROM functional_alltypes t2
  LIMIT 100
) t0
  CROSS JOIN (
    SELECT t2.*
    FROM functional_alltypes t2
    LIMIT 100
  ) t1