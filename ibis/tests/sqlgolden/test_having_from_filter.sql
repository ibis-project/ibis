SELECT `b`, sum(`a`) AS `sum`
FROM t
WHERE `b` = 'm'
GROUP BY 1
HAVING max(`a`) = 2
