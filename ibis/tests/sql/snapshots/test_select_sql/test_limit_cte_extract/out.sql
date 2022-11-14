WITH t0 AS (
  SELECT *
  FROM functional_alltypes
  LIMIT 100
)
SELECT t0.*
FROM t0
  INNER JOIN t0 t1