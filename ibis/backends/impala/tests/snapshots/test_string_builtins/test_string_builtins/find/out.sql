SELECT
  LOCATE('a', `t0`.`string_col`) - 1 AS `StringFind(string_col, 'a')`
FROM `functional_alltypes` AS `t0`