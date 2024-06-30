SELECT
  COALESCE(`t0`.`string_col`, 'foo') AS `Coalesce((string_col, 'foo'))`
FROM `functional_alltypes` AS `t0`