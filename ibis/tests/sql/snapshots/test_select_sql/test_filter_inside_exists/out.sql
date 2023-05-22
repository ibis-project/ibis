WITH t0 AS (
  SELECT t2.*
  FROM purchases t2
  WHERE t2.`ts` > '2015-08-15'
)
SELECT t1.*
FROM events t1
WHERE EXISTS (
  SELECT 1
  FROM t0
  WHERE t1.`user_id` = t0.`user_id`
)