SELECT
  BIT_OR(IF(t0.`bigint_col` > 0, t0.`int_col`, NULL)) AS `BitOr_int_col_ Greater_bigint_col_ 0`
FROM functional_alltypes AS t0