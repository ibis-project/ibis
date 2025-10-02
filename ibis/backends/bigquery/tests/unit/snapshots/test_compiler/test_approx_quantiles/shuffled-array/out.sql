SELECT
  [
    APPROX_QUANTILES(`t0`.`double_col`, 4 IGNORE NULLS)[2],
    APPROX_QUANTILES(`t0`.`double_col`, 4 IGNORE NULLS)[1],
    APPROX_QUANTILES(`t0`.`double_col`, 4 IGNORE NULLS)[3]
  ] AS `qs`
FROM `functional_alltypes` AS `t0`