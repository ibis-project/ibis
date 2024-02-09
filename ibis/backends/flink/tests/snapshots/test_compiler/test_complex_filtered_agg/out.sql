SELECT
  `t0`.`b`,
  COUNT(*) AS `total`,
  AVG(`t0`.`a`) AS `avg_a`,
  AVG(CASE WHEN `t0`.`g` = 'A' THEN `t0`.`a` END) AS `avg_a_A`,
  AVG(CASE WHEN `t0`.`g` = 'B' THEN `t0`.`a` END) AS `avg_a_B`
FROM `table` AS `t0`
GROUP BY
  `t0`.`b`