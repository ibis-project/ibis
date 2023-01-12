SELECT BIT_XOR(if(t0.`bigint_col` > 0, t0.`int_col`, NULL)) AS `tmp`
FROM functional_alltypes t0