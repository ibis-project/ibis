SELECT
  NTILE(3) OVER (ORDER BY `t0`.`double_col` ASC NULLS LAST) - 1 AS `NTile(3)`
FROM `functional_alltypes` AS `t0`