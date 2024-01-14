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
  (
    ROW_NUMBER() OVER (ORDER BY `t0`.`f` ASC NULLS LAST) - 1
  ) / 2 AS `new`
FROM `alltypes` AS `t0`