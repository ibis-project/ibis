SELECT
  LAG(`t0`.`string_col`) OVER (ORDER BY NULL ASC NULLS LAST) AS `Lag(string_col)`
FROM `functional_alltypes` AS `t0`