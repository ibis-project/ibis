SELECT `b` AS `b`, count(1) AS `count`
FROM (
  SELECT `b` AS `b`
  FROM (
    SELECT *
    FROM t
    ORDER BY `b` ASC
  ) t1
) t0
GROUP BY 1