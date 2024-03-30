SELECT
  SUBSTRING(`t0`.`string_col`, IF((
    0 + 1
  ) >= 1, 0 + 1, 0 + 1 + LENGTH(`t0`.`string_col`)), 3) AS `Substring(string_col, 0, 3)`
FROM `functional_alltypes` AS `t0`