SELECT t0.*
FROM foo t0
WHERE t0.`y` > (
  SELECT avg(t1.`y`) AS `mean`
  FROM foo t1
  WHERE t0.`dept_id` = t1.`dept_id`
)
