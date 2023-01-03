SELECT BIT_XOR(if(`bigint_col` > 0, `int_col`, NULL)) AS `tmp`
FROM functional_alltypes