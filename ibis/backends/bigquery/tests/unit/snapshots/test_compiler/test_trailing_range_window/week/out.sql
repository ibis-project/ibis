SELECT *,
       avg(`float_col`) OVER (ORDER BY UNIX_MICROS(`timestamp_col`) RANGE BETWEEN 604800000000 PRECEDING AND CURRENT ROW) AS `win_avg`
FROM functional_alltypes