SELECT t0.`g`, sum(t0.`b`) AS `b_sum`
FROM table t0
GROUP BY t0.`g`
HAVING count(*) >= 1000