SELECT
  t0.*,
  avg(t0.`float_col`) OVER (PARTITION BY t0.`year` ORDER BY t0.`month` ASC RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS `two_month_avg`
FROM functional_alltypes AS t0