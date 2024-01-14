SELECT
  `t1`.`a`,
  `t1`.`b`
FROM `t` AS `t1`
INNER JOIN `t` AS `t3`
  ON `t1`.`a` = `t3`.`a`
  AND (
    (
      `t1`.`a` <> `t3`.`b` OR `t1`.`b` <> `t3`.`a`
    )
    AND NOT (
      `t1`.`a` <> `t3`.`b` AND `t1`.`b` <> `t3`.`a`
    )
  )