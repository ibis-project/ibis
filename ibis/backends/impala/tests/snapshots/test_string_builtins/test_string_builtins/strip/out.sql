SELECT
  RTRIM(LTRIM(`t0`.`string_col`, ' \t\n\r\v\f'), ' \t\n\r\v\f') AS `Strip(string_col)`
FROM `functional_alltypes` AS `t0`