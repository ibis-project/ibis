SELECT
  `t0`.`string_col` RLIKE '[\\d]+' AS `RegexSearch(string_col, '[\\d]+')`
FROM `functional_alltypes` AS `t0`