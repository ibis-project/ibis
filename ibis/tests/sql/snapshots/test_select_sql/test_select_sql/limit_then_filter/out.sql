SELECT t0.*
FROM (
  SELECT t1.*
  FROM star1 t1
  LIMIT 10
) t0
WHERE t0.`f` > 0