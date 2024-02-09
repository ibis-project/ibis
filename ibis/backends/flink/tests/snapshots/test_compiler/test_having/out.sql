SELECT
  `t1`.`g`,
  `t1`.`b_sum`
FROM (
  SELECT
    `t0`.`g`,
    SUM(`t0`.`b`) AS `b_sum`,
    COUNT(*) AS `CountStar(table)`
  FROM `table` AS `t0`
  GROUP BY
    `t0`.`g`
) AS `t1`
WHERE
  `t1`.`CountStar(table)` >= 1000