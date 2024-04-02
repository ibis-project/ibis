SELECT
  CONCAT(
    UPPER(
      SUBSTRING(`t0`.`string_col`, IF((
        0 + 1
      ) >= 1, 0 + 1, 0 + 1 + LENGTH(`t0`.`string_col`)), 1)
    ),
    LOWER(
      SUBSTRING(
        `t0`.`string_col`,
        IF((
          1 + 1
        ) >= 1, 1 + 1, 1 + 1 + LENGTH(`t0`.`string_col`)),
        LENGTH(`t0`.`string_col`)
      )
    )
  ) AS `Capitalize(string_col)`
FROM `functional_alltypes` AS `t0`