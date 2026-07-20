SELECT
  ARRAY_AGG(`t0`.`string_col` IGNORE NULLS ORDER BY `t0`.`id` DESC
  LIMIT 5) AS `ArrayCollect_string_col_5__id`
FROM `functional_alltypes` AS `t0`