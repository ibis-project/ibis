SELECT
  `t0`.`id`,
  `t0`.`bool_col`,
  `t0`.`tinyint_col`,
  `t0`.`smallint_col`,
  `t0`.`int_col`,
  `t0`.`bigint_col`,
  `t0`.`float_col`,
  `t0`.`double_col`,
  `t0`.`date_string_col`,
  `t0`.`string_col`,
  `t0`.`timestamp_col`,
  `t0`.`year`,
  `t0`.`month`,
  AVG(`t0`.`float_col`) OVER (ORDER BY `t0`.`timestamp_col` ASC RANGE BETWEEN INTERVAL '1' WEEK preceding AND INTERVAL 0 WEEK following) AS `win_avg`
FROM `functional_alltypes` AS `t0`