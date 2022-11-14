SELECT sum(if(`g` IS NULL, 1, 0)) AS `sum`
FROM alltypes