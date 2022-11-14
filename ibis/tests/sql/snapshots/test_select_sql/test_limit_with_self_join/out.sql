SELECT count(1) AS `count`
FROM (
  SELECT t1.`id` AS `id_x`, t1.`bool_col` AS `bool_col_x`,
         t1.`tinyint_col` AS `tinyint_col_x`,
         t1.`smallint_col` AS `smallint_col_x`,
         t1.`int_col` AS `int_col_x`, t1.`bigint_col` AS `bigint_col_x`,
         t1.`float_col` AS `float_col_x`,
         t1.`double_col` AS `double_col_x`,
         t1.`date_string_col` AS `date_string_col_x`,
         t1.`string_col` AS `string_col_x`,
         t1.`timestamp_col` AS `timestamp_col_x`, t1.`year` AS `year_x`,
         t1.`month` AS `month_x`, t2.`id` AS `id_y`,
         t2.`bool_col` AS `bool_col_y`,
         t2.`tinyint_col` AS `tinyint_col_y`,
         t2.`smallint_col` AS `smallint_col_y`,
         t2.`int_col` AS `int_col_y`, t2.`bigint_col` AS `bigint_col_y`,
         t2.`float_col` AS `float_col_y`,
         t2.`double_col` AS `double_col_y`,
         t2.`date_string_col` AS `date_string_col_y`,
         t2.`string_col` AS `string_col_y`,
         t2.`timestamp_col` AS `timestamp_col_y`, t2.`year` AS `year_y`,
         t2.`month` AS `month_y`
  FROM functional_alltypes t1
    INNER JOIN functional_alltypes t2
      ON t1.`tinyint_col` < extract(t2.`timestamp_col`, 'minute')
) t0