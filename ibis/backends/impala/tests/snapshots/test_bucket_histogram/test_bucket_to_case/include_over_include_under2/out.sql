SELECT
  CAST(CASE WHEN `t0`.`f` < 10 THEN 0 WHEN 10 <= `t0`.`f` THEN 1 ELSE NULL END AS DOUBLE) AS `Cast(Bucket(f, ()), float64)`
FROM `alltypes` AS `t0`