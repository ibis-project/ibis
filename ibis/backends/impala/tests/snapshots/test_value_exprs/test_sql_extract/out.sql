SELECT
  EXTRACT(year FROM `t0`.`i`) AS `year`,
  EXTRACT(month FROM `t0`.`i`) AS `month`,
  EXTRACT(day FROM `t0`.`i`) AS `day`
FROM `alltypes` AS `t0`