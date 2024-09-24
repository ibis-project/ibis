SELECT
  *
FROM (
  SELECT
    *
  FROM `test` AS `t0`
  WHERE
    `t0`.`x` > 10
) AS `t1`
WHERE
  RAND(UTC_TO_UNIX_MICROS(UTC_TIMESTAMP())) <= 0.5