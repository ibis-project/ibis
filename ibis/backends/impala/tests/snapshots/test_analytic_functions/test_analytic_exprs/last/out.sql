SELECT
  LAST_VALUE(`t0`.`double_col`) OVER (ORDER BY `t0`.`id` ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS `Last(double_col, ())`
FROM `functional_alltypes` AS `t0`