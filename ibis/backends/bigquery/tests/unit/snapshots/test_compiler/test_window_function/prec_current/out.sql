SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ASC ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS `win_avg`
FROM functional_alltypes