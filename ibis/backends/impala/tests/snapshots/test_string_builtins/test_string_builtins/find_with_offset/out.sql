SELECT
  LOCATE('a', `t0`.`string_col`, 2 + 1) - 1 AS `StringFind(string_col, 'a', 2)`
FROM `functional_alltypes` AS `t0`