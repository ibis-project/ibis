SELECT APPROX_QUANTILES(if(t0.`month` > 0, t0.`double_col`, NULL), 2)[OFFSET(1)] AS `tmp`
FROM functional_alltypes t0