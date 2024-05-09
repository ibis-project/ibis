SELECT
  AVG(IF(`t0`.`month` > 6, CAST(`t0`.`bool_col` AS INT64), NULL)) AS `Mean_bool_col_Greater_month_6`
FROM `functional_alltypes` AS `t0`