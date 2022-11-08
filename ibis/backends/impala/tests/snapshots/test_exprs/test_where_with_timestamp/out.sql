SELECT `uuid`, min(if(`search_level` = 1, `ts`, NULL)) AS `min_date`
FROM t
GROUP BY 1