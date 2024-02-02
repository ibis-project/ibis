SELECT
  IF(
    3 >= 0,
    substr(`t0`.`value`, 3 + 1, 1),
    substr(`t0`.`value`, LENGTH(`t0`.`value`) + 3 + 1, 1)
  ) AS `tmp`
FROM `t` AS `t0`