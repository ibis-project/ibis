SELECT
  `t0`.`a`,
  `t0`.`b`,
  `t0`.`c`,
  `t0`.`d`,
  `t0`.`e`,
  `t0`.`f`,
  `t0`.`g`,
  `t0`.`h`,
  `t0`.`i`,
  `t0`.`j`,
  `t0`.`k`,
  `t0`.`f` / SUM(`t0`.`f`) OVER (ORDER BY NULL ASC NULLS LAST) AS `normed_f`
FROM `alltypes` AS `t0`