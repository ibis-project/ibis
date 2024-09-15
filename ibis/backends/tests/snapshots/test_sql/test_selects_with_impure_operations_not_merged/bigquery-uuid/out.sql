SELECT
  `t1`.`x`,
  `t1`.`y`,
  `t1`.`z`,
  IF(`t1`.`y` = `t1`.`z`, 'big', 'small') AS `size`
FROM (
  SELECT
    `t0`.`x`,
    GENERATE_UUID() AS `y`,
    GENERATE_UUID() AS `z`
  FROM `t` AS `t0`
) AS `t1`