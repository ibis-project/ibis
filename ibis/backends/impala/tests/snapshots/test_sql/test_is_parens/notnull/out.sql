SELECT
  *
FROM `table` AS `t0`
WHERE
  (
    `t0`.`a` IS NOT NULL
  ) = (
    `t0`.`b` IS NOT NULL
  )