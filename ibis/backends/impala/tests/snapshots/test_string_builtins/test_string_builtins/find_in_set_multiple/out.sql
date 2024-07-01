SELECT
  FIND_IN_SET(`t0`.`string_col`, CONCAT_WS(',', 'a', 'b')) - 1 AS `FindInSet(string_col, ('a', 'b'))`
FROM `functional_alltypes` AS `t0`