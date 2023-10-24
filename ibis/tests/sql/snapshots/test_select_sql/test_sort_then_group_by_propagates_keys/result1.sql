SELECT t0.`b`, count(1) AS `b_count`
FROM (
  SELECT t1.`b`
  FROM t t1
  ORDER BY t1.`a` ASC
) t0
GROUP BY 1