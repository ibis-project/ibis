SELECT
  RTRIM(
    LTRIM(`t0`.`string_col`, CONCAT(' \t\n\r', CHR(  11), CHR(  12))),
    CONCAT(' \t\n\r', CHR(  11), CHR(  12))
  ) AS `Strip(string_col)`
FROM `functional_alltypes` AS `t0`