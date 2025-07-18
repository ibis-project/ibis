SELECT
  BIT_XOR(IF(`t0`.`bigint_col` > 0, `t0`.`int_col`, NULL)) AS `BitXor_int_col_Greater_bigint_col_0`
FROM `functional_alltypes` AS `t0`