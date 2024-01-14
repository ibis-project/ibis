SELECT
  FIND_IN_SET(`t0`.`string_col`, CONCAT_WS(',', 'a')) - 1 AS `FindInSet(string_col)`
FROM `functional_alltypes` AS `t0`