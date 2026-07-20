SELECT
  COALESCE(
    ARRAY_AGG(`t0`.`string_col` IGNORE NULLS ORDER BY `t0`.`id` DESC
    LIMIT 5),
    ARRAY<STRING>[]
  ) AS `ArraySlice_ArrayCollect_string_col__id_0_5`
FROM `functional_alltypes` AS `t0`