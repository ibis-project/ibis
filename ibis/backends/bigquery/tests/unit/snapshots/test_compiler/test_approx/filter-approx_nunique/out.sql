SELECT APPROX_COUNT_DISTINCT(if(t0.`month` > 0, t0.`double_col`, NULL)) AS `ApproxCountDistinct_double_col_Greater_month`
FROM functional_alltypes t0