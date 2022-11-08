SELECT BIT_AND(if(`bigint_col` > 0, `int_col`, NULL)) AS `bit_and`
FROM functional_alltypes