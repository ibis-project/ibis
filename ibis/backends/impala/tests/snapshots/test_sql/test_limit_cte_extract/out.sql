SELECT
  `t2`.`id`,
  `t2`.`bool_col`,
  `t2`.`tinyint_col`,
  `t2`.`smallint_col`,
  `t2`.`int_col`,
  `t2`.`bigint_col`,
  `t2`.`float_col`,
  `t2`.`double_col`,
  `t2`.`date_string_col`,
  `t2`.`string_col`,
  `t2`.`timestamp_col`,
  `t2`.`year`,
  `t2`.`month`
FROM (
  SELECT
    *
  FROM `functional_alltypes` AS `t0`
  LIMIT 100
) AS `t2`
INNER JOIN (
  SELECT
    *
  FROM `functional_alltypes` AS `t0`
  LIMIT 100
) AS `t4`
  ON TRUE