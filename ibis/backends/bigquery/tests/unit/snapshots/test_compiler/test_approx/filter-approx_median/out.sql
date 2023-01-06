SELECT APPROX_QUANTILES(if(t0.`month` > 0, t0.`double_col`, NULL), 2)[OFFSET(1)] AS `ApproxMedian_double_col_Greater_month`
FROM functional_alltypes t0