SELECT
  APPROX_QUANTILES(`t0`.`double_col`, 4 IGNORE NULLS) AS `qs`
FROM `functional_alltypes` AS `t0`