SELECT
  CONCAT(
    UPPER(
      IF(
        (
          0 + 1
        ) >= 1,
        SUBSTRING(`t0`.`string_col`, 0 + 1, 1),
        SUBSTRING(`t0`.`string_col`, 0 + 1 + LENGTH(`t0`.`string_col`), 1)
      )
    ),
    LOWER(
      IF(
        (
          1 + 1
        ) >= 1,
        SUBSTRING(`t0`.`string_col`, 1 + 1, LENGTH(`t0`.`string_col`)),
        SUBSTRING(`t0`.`string_col`, 1 + 1 + LENGTH(`t0`.`string_col`), LENGTH(`t0`.`string_col`))
      )
    )
  ) AS `Capitalize(string_col)`
FROM `functional_alltypes` AS `t0`