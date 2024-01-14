SELECT
  PERCENT_RANK() OVER (ORDER BY `t0`.`double_col` ASC NULLS LAST) AS `PercentRank()`
FROM `functional_alltypes` AS `t0`