SELECT APPROX_COUNT_DISTINCT(if(t0.`month` > 0, t0.`double_col`, NULL)) AS `tmp`
FROM functional_alltypes t0