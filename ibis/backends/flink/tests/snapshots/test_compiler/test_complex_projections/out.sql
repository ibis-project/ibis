SELECT
  `t1`.`a`,
  AVG(ABS(`t1`.`the_sum`)) AS `mad`
FROM (
  SELECT
    `t0`.`a`,
    `t0`.`c`,
    SUM(`t0`.`b`) AS `the_sum`
  FROM `table` AS `t0`
  GROUP BY
    `t0`.`a`,
    `t0`.`c`
) AS `t1`
GROUP BY
  `t1`.`a`