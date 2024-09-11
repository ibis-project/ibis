SELECT
  LTRIM(`t0`.`string_col`, ' \t\n\r\v\f') AS `LStrip(string_col)`
FROM `functional_alltypes` AS `t0`