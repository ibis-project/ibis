SELECT
  `t0`.`a`,
  9 AS `i`,
  'foo' AS `s`
FROM `test` AS `t0`
ORDER BY
  CASE WHEN `t0`.`a` IS NULL THEN 1 ELSE 0 END, `t0`.`a` ASC