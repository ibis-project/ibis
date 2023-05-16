SELECT t0.`uuid`, min(if(t0.`search_level` = 1, t0.`ts`, NULL)) AS `min_date`
FROM `t` t0
GROUP BY 1