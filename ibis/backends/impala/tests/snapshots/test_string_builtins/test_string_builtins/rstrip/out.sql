SELECT
  REGEXP_REPLACE(`t0`.`string_col`, '\\s+$', '') AS `RStrip(string_col)`
FROM `functional_alltypes` AS `t0`