SELECT
  LAG(`t0`.`string_col`, 2) OVER (ORDER BY NULL ASC NULLS LAST) AS `Lag(string_col, 2)`
FROM `functional_alltypes` AS `t0`