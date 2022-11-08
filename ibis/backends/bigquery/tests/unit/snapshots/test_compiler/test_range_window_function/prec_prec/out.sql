SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY UNIX_MICROS(`timestamp_col`) RANGE BETWEEN 4 PRECEDING AND 2 PRECEDING) AS `two_month_avg`
FROM functional_alltypes