SELECT
  LOCATE('a', `t0`.`string_col`, `t0`.`tinyint_col` + 1) - 1 AS `StringFind(string_col, 'a', tinyint_col)`
FROM `functional_alltypes` AS `t0`