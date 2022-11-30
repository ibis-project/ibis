SELECT avg(if(`month` > 6, CAST(`bool_col` AS INT64), NULL)) AS `tmp`
FROM functional_alltypes