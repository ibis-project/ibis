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
  LAG(`t0`.`f`) OVER (PARTITION BY `t0`.`g` ORDER BY NULL ASC NULLS LAST) AS `lag`,
  LEAD(`t0`.`f`) OVER (PARTITION BY `t0`.`g` ORDER BY NULL ASC NULLS LAST) - `t0`.`f` AS `fwd_diff`,
  FIRST_VALUE(`t0`.`f`) OVER (PARTITION BY `t0`.`g`) AS `first`,
  LAST_VALUE(`t0`.`f`) OVER (PARTITION BY `t0`.`g`) AS `last`,
  LAG(`t0`.`f`) OVER (PARTITION BY `t0`.`g` ORDER BY `t0`.`d` ASC NULLS LAST) AS `lag2`
FROM `alltypes` AS `t0`