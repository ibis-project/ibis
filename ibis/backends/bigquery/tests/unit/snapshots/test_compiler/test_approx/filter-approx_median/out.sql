SELECT
  APPROX_QUANTILES(IF(t0.`month` > 0, t0.`double_col`, NULL), 2)[OFFSET(1)] AS `ApproxMedian_double_col_ Greater_month_ 0`
FROM functional_alltypes AS t0