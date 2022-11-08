SELECT *,
       avg(`float_col`) OVER (PARTITION BY `year` ORDER BY `month` ASC RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS `two_month_avg`
FROM functional_alltypes