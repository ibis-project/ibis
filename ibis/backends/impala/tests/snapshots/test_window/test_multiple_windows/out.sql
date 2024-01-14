SELECT
  `t0`.`g`,
  SUM(`t0`.`f`) OVER (PARTITION BY `t0`.`g` ORDER BY NULL ASC NULLS LAST) - SUM(`t0`.`f`) OVER (ORDER BY NULL ASC NULLS LAST) AS `result`
FROM `alltypes` AS `t0`