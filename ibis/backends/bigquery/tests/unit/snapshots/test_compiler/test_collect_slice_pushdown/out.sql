SELECT
  COALESCE(
    ARRAY_AGG(DISTINCT `t0`.`string_col` IGNORE NULLS ORDER BY `t0`.`string_col` DESC
    LIMIT 5),
    ARRAY<STRING>[]
  ) AS `ArraySlice_ArrayCollect_string_col__string_col_0_5`
FROM `functional_alltypes` AS `t0`