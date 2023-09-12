SELECT t0.*
FROM (
  SELECT t1.*, t1.`f` - t2.`value1` AS `diff`
  FROM star1 t1
    INNER JOIN star2 t2
      ON t1.`foo_id` = t2.`foo_id`
) t0
WHERE t0.`diff` > 1