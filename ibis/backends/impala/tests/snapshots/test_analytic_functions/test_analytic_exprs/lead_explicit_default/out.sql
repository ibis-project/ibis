SELECT
  LEAD(`t0`.`string_col`, 1, 0) OVER (ORDER BY NULL ASC NULLS LAST) AS `Lead(string_col, 0)`
FROM `functional_alltypes` AS `t0`