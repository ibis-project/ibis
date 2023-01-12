SELECT sum(if((t0.`month` > 6) AND (t0.`month` < 10), CAST(t0.`bool_col` AS INT64), NULL)) AS `tmp`
FROM functional_alltypes t0