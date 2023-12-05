SELECT t0.`foo_id`, sum(t0.`value1`) AS `total`
FROM (
  SELECT t1.*, t2.`value1`
  FROM star1 t1
    INNER JOIN star2 t2
      ON t1.`foo_id` = t2.`foo_id`
) t0
GROUP BY 1