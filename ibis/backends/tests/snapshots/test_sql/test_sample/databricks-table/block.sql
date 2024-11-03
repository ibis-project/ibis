SELECT
  *
FROM (
  SELECT
    *
  FROM `test` AS `t0`
  WHERE
    `t0`.`x` > 10
) TABLESAMPLE (50.0 PERCENT) AS `t1`