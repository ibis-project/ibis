SELECT
  `t0`.`one`,
  `t0`.`two`,
  `t0`.`three`,
  SUM(`t0`.`two`) OVER (PARTITION BY `t0`.`three` ORDER BY `t0`.`one` ASC NULLS LAST) AS `four`
FROM `my_data` AS `t0`