SELECT APPROX_QUANTILES(`double_col`, 2)[OFFSET(1)] AS `approx_median`
FROM functional_alltypes