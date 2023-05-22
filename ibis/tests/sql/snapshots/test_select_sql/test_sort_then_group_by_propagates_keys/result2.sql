WITH t0 AS (
  SELECT t2.*
  FROM t t2
  ORDER BY t2.`b` ASC
)
SELECT t1.`b`, count(1) AS `b_count`
FROM (
  SELECT t0.`b`
  FROM t0
) t1
GROUP BY 1