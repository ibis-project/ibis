SELECT
  EXTRACT(millisecond FROM `t0`.`i`) % 1000 AS `ExtractMillisecond(i)`
FROM `alltypes` AS `t0`