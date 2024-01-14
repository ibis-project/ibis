SELECT
  IF(
    (
      0 + 1
    ) >= 1,
    SUBSTRING(`t0`.`string_col`, 0 + 1, 3),
    SUBSTRING(`t0`.`string_col`, 0 + 1 + LENGTH(`t0`.`string_col`), 3)
  ) AS `Substring(string_col, 0, 3)`
FROM `functional_alltypes` AS `t0`