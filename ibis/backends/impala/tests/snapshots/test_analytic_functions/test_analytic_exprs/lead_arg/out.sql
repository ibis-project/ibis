SELECT
  LEAD(`t0`.`string_col`, 2) OVER (ORDER BY NULL ASC) AS `Lead(string_col, 2)`
FROM `functional_alltypes` AS `t0`