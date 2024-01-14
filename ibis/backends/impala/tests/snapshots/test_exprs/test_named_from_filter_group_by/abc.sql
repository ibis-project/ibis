SELECT
  `t1`.`key`,
  SUM((
    (
      `t1`.`value` + 1
    ) + 2
  ) + 3) AS `abc`
FROM (
  SELECT
    `t0`.`key`,
    `t0`.`value`
  FROM `t0` AS `t0`
  WHERE
    `t0`.`value` = 42
) AS `t1`
GROUP BY
  1