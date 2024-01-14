SELECT
  EXTRACT(microsecond FROM `t0`.`i`) % 1000000 AS `ExtractMicrosecond(i)`
FROM `alltypes` AS `t0`