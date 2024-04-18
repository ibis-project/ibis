SELECT
  `t1`.`a`,
  `t1`.`b`
FROM `t` AS `t1`
INNER JOIN `t` AS `t2`
  ON `t1`.`a` = `t2`.`a`
  AND (
    (
      `t1`.`a` <> `t2`.`b`
    ) OR (
      `t1`.`b` <> `t2`.`a`
    )
  )