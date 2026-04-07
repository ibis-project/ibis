SELECT
  REGEXP_REPLACE(`t0`.`string_col`, '^\\s+|\\s+$', '') AS `Strip(string_col)`
FROM `functional_alltypes` AS `t0`