SELECT
  `t0`.`a`,
  `t0`.`b`
FROM `t0` AS `t0`
WHERE
  `t0`.`a` < 100
  AND `t0`.`a` = (
    SELECT
      MAX(`t1`.`a`) AS `Max(a)`
    FROM (
      SELECT
        `t0`.`a`,
        `t0`.`b`
      FROM `t0` AS `t0`
      WHERE
        `t0`.`a` < 100
    ) AS `t1`
  )
  AND `t0`.`b` = 'a'