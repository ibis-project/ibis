SELECT sum(if((`month` > 6) AND (`month` < 10), CAST(`bool_col` AS INT64), NULL)) AS `sum`
FROM functional_alltypes