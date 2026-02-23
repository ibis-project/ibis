INSERT OVERWRITE `foo`.`testing123456` (`id`, `bool_col`, `tinyint_col`, `smallint_col`, `int_col`, `bigint_col`, `float_col`, `double_col`, `date_string_col`, `string_col`, `timestamp_col`, `year`, `month`) 
SELECT
  *
FROM `functional_alltypes` AS `t0`
LIMIT 10