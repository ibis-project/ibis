WITH t0 AS (
  SELECT t2.*
  FROM star1 t2
  LIMIT 100
)
SELECT t0.*
FROM t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`