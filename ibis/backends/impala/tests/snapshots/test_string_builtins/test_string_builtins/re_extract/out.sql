SELECT
  REGEXP_EXTRACT(`t0`.`string_col`, '[\\d]+', 0) AS `RegexExtract(string_col, '[\\d]+', 0)`
FROM `functional_alltypes` AS `t0`