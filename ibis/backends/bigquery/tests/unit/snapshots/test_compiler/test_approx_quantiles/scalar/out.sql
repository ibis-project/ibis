SELECT
  APPROX_QUANTILES(`t0`.`double_col`, 2 IGNORE NULLS)[1] AS `qs`
FROM `functional_alltypes` AS `t0`