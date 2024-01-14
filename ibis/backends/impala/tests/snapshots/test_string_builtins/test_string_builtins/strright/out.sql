SELECT
  SUBSTRING(`t0`.`string_col`, LENGTH(`t0`.`string_col`) - (
    4 - 1
  )) AS `StrRight(string_col, 4)`
FROM `functional_alltypes` AS `t0`