SELECT t0.*
FROM events t0
WHERE EXISTS (
  SELECT 1
  FROM (
    SELECT t2.*
    FROM purchases t2
    WHERE t2.`ts` > '2015-08-15'
  ) t1
  WHERE t0.`user_id` = t1.`user_id`
)