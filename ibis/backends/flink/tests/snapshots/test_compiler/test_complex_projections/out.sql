SELECT t0.`a`, avg(abs(t0.`the_sum`)) AS `mad`
FROM (
  SELECT t1.`a`, t1.`c`, sum(t1.`b`) AS `the_sum`
  FROM table t1
  GROUP BY t1.`a`, t1.`c`
) t0
GROUP BY t0.`a`