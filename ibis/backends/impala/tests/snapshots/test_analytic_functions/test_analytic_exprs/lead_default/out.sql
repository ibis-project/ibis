SELECT
  LEAD(`t0`.`string_col`) OVER (ORDER BY NULL ASC) AS `Lead(string_col)`
FROM `functional_alltypes` AS `t0`