SELECT BIT_OR(if(`bigint_col` > 0, `int_col`, NULL)) AS `bit_or`
FROM functional_alltypes