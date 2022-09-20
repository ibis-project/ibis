SELECT t0.*, t1.`value1`
FROM (
  SELECT `foo_id`, sum(`f`) AS `total`
  FROM star1
  GROUP BY 1
) t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`
