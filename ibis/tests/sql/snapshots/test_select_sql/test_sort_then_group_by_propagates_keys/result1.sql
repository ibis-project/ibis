SELECT `b`, count(1) AS `count`
FROM (
  SELECT `b`
  FROM (
    SELECT *
    FROM t
    ORDER BY `a` ASC
  ) t1
) t0
GROUP BY 1