SELECT
  `t0`.`uuid`,
  MIN(IF(`t0`.`search_level` = 1, `t0`.`ts`, NULL)) AS `min_date`
FROM `t` AS `t0`
GROUP BY
  1