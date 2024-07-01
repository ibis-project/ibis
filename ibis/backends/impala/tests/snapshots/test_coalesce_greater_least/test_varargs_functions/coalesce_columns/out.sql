SELECT
  COALESCE(`t0`.`int_col`, `t0`.`bigint_col`) AS `Coalesce((int_col, bigint_col))`
FROM `functional_alltypes` AS `t0`