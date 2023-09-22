SELECT
  t0.*,
  avg(t0.`float_col`) OVER (PARTITION BY t0.`year` ORDER BY UNIX_MICROS(t0.`timestamp_col`) ASC RANGE BETWEEN 4 PRECEDING AND 2 PRECEDING) AS `two_month_avg`
FROM functional_alltypes AS t0