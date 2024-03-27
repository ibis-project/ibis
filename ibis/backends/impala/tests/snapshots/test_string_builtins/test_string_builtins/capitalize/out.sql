SELECT
  CONCAT(
    UPPER(SUBSTRING(`t0`.`string_col`, 1, 1)),
    LOWER(SUBSTRING(`t0`.`string_col`, 2, LENGTH(`t0`.`string_col`)))
  ) AS `Capitalize(string_col)`
FROM `functional_alltypes` AS `t0`