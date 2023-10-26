SELECT t0.*, t0.`f` - t1.`value1` AS `diff`
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`
WHERE (t0.`f` - t1.`value1`) > 1