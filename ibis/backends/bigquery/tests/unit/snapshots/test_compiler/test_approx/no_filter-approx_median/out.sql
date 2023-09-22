SELECT
  APPROX_QUANTILES(t0.`double_col`, 2)[OFFSET(1)] AS `ApproxMedian_double_col`
FROM functional_alltypes AS t0