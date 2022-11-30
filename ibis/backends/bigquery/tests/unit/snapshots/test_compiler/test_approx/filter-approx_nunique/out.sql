SELECT APPROX_COUNT_DISTINCT(if(`month` > 0, `double_col`, NULL)) AS `tmp`
FROM functional_alltypes