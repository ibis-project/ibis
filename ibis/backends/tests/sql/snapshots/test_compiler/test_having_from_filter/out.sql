SELECT t0.`b`, sum(t0.`a`) AS `sum`
FROM t t0
WHERE t0.`b` = 'm'
GROUP BY 1
HAVING max(t0.`a`) = 2