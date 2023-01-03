SELECT APPROX_QUANTILES(`double_col`, 2)[OFFSET(1)] AS `tmp`
FROM functional_alltypes