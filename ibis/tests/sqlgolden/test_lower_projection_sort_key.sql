SELECT *
FROM (
  SELECT t1.*
  FROM (
    SELECT t4.*, t2.`value1`
    FROM (
      SELECT `foo_id`, sum(`f`) AS `total`
      FROM star1
      GROUP BY 1
    ) t4
      INNER JOIN star2 t2
        ON t4.`foo_id` = t2.`foo_id`
  ) t1
  WHERE t1.`total` > 100
) t0
ORDER BY `total` DESC
