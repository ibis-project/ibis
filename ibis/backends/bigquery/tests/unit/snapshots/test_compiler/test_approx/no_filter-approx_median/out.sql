SELECT
  approx_quantiles(`t0`.`double_col`, 2)[offset(1)] AS `ApproxMedian_double_col`
FROM `functional_alltypes` AS `t0`