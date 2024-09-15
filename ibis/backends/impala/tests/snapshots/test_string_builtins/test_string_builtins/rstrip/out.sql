SELECT
  RTRIM(`t0`.`string_col`, ' \t\n\r\v\f') AS `RStrip(string_col)`
FROM `functional_alltypes` AS `t0`