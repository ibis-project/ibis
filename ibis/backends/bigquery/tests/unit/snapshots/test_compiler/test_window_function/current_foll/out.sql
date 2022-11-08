SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ASC ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS `win_avg`
FROM functional_alltypes