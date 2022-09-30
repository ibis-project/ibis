SELECT t0.*
FROM (
  SELECT *
  FROM star1
  LIMIT 100
) t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`
