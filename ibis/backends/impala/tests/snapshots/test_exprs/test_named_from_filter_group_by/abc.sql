SELECT t0.`key`, sum(((t0.`value` + 1) + 2) + 3) AS `abc`
FROM `t0` t0
WHERE t0.`value` = 42
GROUP BY 1