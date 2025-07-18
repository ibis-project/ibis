SELECT
  BIT_AND(IF(`t0`.`bigint_col` > 0, `t0`.`int_col`, NULL)) AS `BitAnd_int_col_Greater_bigint_col_0`
FROM `functional_alltypes` AS `t0`