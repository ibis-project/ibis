SELECT
  avg(IF(t0.`month` > 6, CAST(t0.`bool_col` AS INT64), NULL)) AS `Mean_bool_col_ Greater_month_ 6`
FROM functional_alltypes AS t0