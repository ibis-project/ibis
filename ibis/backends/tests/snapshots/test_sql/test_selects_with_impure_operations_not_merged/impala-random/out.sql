SELECT
  `t1`.`x`,
  `t1`.`y`,
  `t1`.`z`,
  IF(`t1`.`y` = `t1`.`z`, 'big', 'small') AS `size`
FROM (
  SELECT
    `t0`.`x`,
    RAND(UTC_TO_UNIX_MICROS(UTC_TIMESTAMP())) AS `y`,
    RAND(UTC_TO_UNIX_MICROS(UTC_TIMESTAMP())) AS `z`
  FROM `t` AS `t0`
) AS `t1`