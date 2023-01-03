SELECT APPROX_QUANTILES(if(`month` > 0, `double_col`, NULL), 2)[OFFSET(1)] AS `tmp`
FROM functional_alltypes