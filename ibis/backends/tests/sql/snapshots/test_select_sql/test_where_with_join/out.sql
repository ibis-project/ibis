SELECT t0.*, t1.`value1`, t1.`value3`
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`
WHERE (t0.`f` > 0) AND
      (t1.`value3` < 1000)