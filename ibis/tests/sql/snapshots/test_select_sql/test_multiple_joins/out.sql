SELECT *, `value1`, t1.`value2`
FROM (
  SELECT t2.`c`, t2.`f`, t2.`foo_id`, t2.`bar_id`,
         t3.`foo_id` AS `foo_id_right`, t3.`value1`, t3.`value3`
  FROM star1 t2
    LEFT OUTER JOIN star2 t3
      ON t2.`foo_id` = t3.`foo_id`
) t0
  INNER JOIN star3 t1
    ON `bar_id` = t1.`bar_id`