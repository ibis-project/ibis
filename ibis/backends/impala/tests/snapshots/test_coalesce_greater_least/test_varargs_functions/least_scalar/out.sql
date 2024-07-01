SELECT
  LEAST(`t0`.`string_col`, 'foo') AS `Least((string_col, 'foo'))`
FROM `functional_alltypes` AS `t0`