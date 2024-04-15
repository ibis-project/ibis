SELECT
  `t1`.`x`,
  `t1`.`y`,
  `t1`.`z`,
  IF(`t1`.`y` = `t1`.`z`, 'big', 'small') AS `size`
FROM (
  SELECT
    `t0`.`x`,
    generate_uuid() AS `y`,
    generate_uuid() AS `z`
  FROM `t` AS `t0`
) AS `t1`