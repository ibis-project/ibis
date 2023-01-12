SELECT t0.`b`, count(1) AS `count`
FROM (
  SELECT t1.`b`
  FROM (
    SELECT t2.*
    FROM t t2
    ORDER BY t2.`b` ASC
  ) t1
) t0
GROUP BY 1