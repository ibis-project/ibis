WITH t0 AS (
  SELECT t1.*
  FROM star1 t1
  LIMIT 10
)
SELECT t0.*
FROM t0
WHERE t0.`f` > 0