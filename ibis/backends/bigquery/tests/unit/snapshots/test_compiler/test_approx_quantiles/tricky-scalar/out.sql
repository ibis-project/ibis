SELECT
  approx_quantiles(`t0`.`double_col`, 100000 IGNORE NULLS)[33333] AS `qs`
FROM `functional_alltypes` AS `t0`