SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `timestamp_col` ASC ROWS BETWEEN 4 PRECEDING AND 2 PRECEDING) AS `win_avg`
FROM functional_alltypes