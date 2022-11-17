SELECT `b`, count(1) AS `count`
FROM t
GROUP BY 1
ORDER BY `b` ASC