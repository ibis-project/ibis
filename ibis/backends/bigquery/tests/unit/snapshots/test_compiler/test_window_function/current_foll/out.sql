SELECT
  t0.*,
  avg(t0.`float_col`) OVER (PARTITION BY t0.`year` ORDER BY t0.`timestamp_col` ASC ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS `win_avg`
FROM functional_alltypes AS t0