SELECT sum(if((t0.`month` > 6) AND (t0.`month` < 10), CAST(t0.`bool_col` AS INT64), NULL)) AS `Sum_bool_col_And_Greater_month_6_Less_month_10_`
FROM functional_alltypes t0