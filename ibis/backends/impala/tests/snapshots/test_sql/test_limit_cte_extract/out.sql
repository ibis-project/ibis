WITH `t1` AS (
  SELECT
    *
  FROM `functional_alltypes` AS `t0`
  LIMIT 100
)
SELECT
  `t3`.`id`,
  `t3`.`bool_col`,
  `t3`.`tinyint_col`,
  `t3`.`smallint_col`,
  `t3`.`int_col`,
  `t3`.`bigint_col`,
  `t3`.`float_col`,
  `t3`.`double_col`,
  `t3`.`date_string_col`,
  `t3`.`string_col`,
  `t3`.`timestamp_col`,
  `t3`.`year`,
  `t3`.`month`
FROM `t1` AS `t3`
INNER JOIN `t1` AS `t5`
  ON TRUE