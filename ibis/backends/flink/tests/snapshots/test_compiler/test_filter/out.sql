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
  `t0`.`k`
FROM `table` AS `t0`
WHERE
  (
    (
      `t0`.`c` > 0
    ) OR (
      `t0`.`c` < 0
    )
  ) AND `t0`.`g` IN ('A', 'B')