SELECT
  [
    approx_quantiles(`t0`.`double_col`, 4 IGNORE NULLS)[1],
    approx_quantiles(`t0`.`double_col`, 4 IGNORE NULLS)[2],
    approx_quantiles(`t0`.`double_col`, 4 IGNORE NULLS)[3]
  ] AS `qs`
FROM `functional_alltypes` AS `t0`