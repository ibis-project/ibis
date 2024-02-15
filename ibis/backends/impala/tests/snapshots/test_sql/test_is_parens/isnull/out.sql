SELECT
  `t0`.`a`,
  `t0`.`b`
FROM `table` AS `t0`
WHERE
  (
    `t0`.`a` IS NULL
  ) = (
    `t0`.`b` IS NULL
  )