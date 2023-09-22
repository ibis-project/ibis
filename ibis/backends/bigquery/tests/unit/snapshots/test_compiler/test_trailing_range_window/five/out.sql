SELECT
  t0.*,
  avg(t0.`float_col`) OVER (ORDER BY UNIX_MICROS(t0.`timestamp_col`) ASC RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) AS `win_avg`
FROM functional_alltypes AS t0