WITH t0 AS (
  SELECT t2.*
  FROM `functional_alltypes` t2
  LIMIT 100
)
SELECT t0.*
FROM t0
  CROSS JOIN t0 t1