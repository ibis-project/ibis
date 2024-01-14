SELECT
  VARIANCE(IF(`t0`.`bigint_col` < 70, `t0`.`double_col`, NULL)) AS `Variance(double_col, Less(bigint_col, 70))`
FROM `functional_alltypes` AS `t0`