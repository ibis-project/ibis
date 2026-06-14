SELECT
  RTRIM(`t0`.`string_col`, CONCAT(' \t\n\r', CHR(  11), CHR(  12))) AS `RStrip(string_col)`
FROM `functional_alltypes` AS `t0`