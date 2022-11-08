SELECT BIT_XOR(if(`bigint_col` > 0, `int_col`, NULL)) AS `bit_xor`
FROM functional_alltypes