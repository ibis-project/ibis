SELECT
  IF(
    (
      2 + 1
    ) >= 1,
    SUBSTRING(`t0`.`string_col`, 2 + 1),
    SUBSTRING(`t0`.`string_col`, 2 + 1 + LENGTH(`t0`.`string_col`))
  ) AS `Substring(string_col, 2)`
FROM `functional_alltypes` AS `t0`