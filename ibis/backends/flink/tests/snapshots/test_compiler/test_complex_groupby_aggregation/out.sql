SELECT
  EXTRACT(year FROM `t0`.`i`) AS `year`,
  EXTRACT(month FROM `t0`.`i`) AS `month`,
  COUNT(*) AS `total`,
  COUNT(DISTINCT `t0`.`b`) AS `b_unique`
FROM `table` AS `t0`
GROUP BY
  EXTRACT(year FROM `t0`.`i`),
  EXTRACT(month FROM `t0`.`i`)