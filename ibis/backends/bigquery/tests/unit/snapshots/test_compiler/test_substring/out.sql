SELECT
  SUBSTRING(`t0`.`value`, IF((
    3 + 1
  ) >= 1, 3 + 1, 3 + 1 + LENGTH(`t0`.`value`)), 1) AS `tmp`
FROM `t` AS `t0`