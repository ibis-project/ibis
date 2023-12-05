SELECT count(1) AS `CountStar()`
FROM (
  SELECT t1.`id`, t1.`bool_col`, t1.`tinyint_col`, t1.`smallint_col`,
         t1.`int_col`, t1.`bigint_col`, t1.`float_col`, t1.`double_col`,
         t1.`date_string_col`, t1.`string_col`, t1.`timestamp_col`,
         t1.`year`, t1.`month`, t2.`id` AS `id_right`,
         t2.`bool_col` AS `bool_col_right`,
         t2.`tinyint_col` AS `tinyint_col_right`,
         t2.`smallint_col` AS `smallint_col_right`,
         t2.`int_col` AS `int_col_right`,
         t2.`bigint_col` AS `bigint_col_right`,
         t2.`float_col` AS `float_col_right`,
         t2.`double_col` AS `double_col_right`,
         t2.`date_string_col` AS `date_string_col_right`,
         t2.`string_col` AS `string_col_right`,
         t2.`timestamp_col` AS `timestamp_col_right`,
         t2.`year` AS `year_right`, t2.`month` AS `month_right`
  FROM functional_alltypes t1
    INNER JOIN functional_alltypes t2
      ON t1.`tinyint_col` < extract(t2.`timestamp_col`, 'minute')
) t0