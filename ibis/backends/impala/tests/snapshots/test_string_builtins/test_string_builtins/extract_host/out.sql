SELECT
  PARSE_URL(`t0`.`string_col`, 'HOST') AS `ExtractHost(string_col)`
FROM `functional_alltypes` AS `t0`