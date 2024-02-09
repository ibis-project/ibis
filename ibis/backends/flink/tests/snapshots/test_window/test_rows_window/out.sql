SELECT
  SUM(`t0`.`f`) OVER (ORDER BY `t0`.`f` ASC NULLS LAST ROWS BETWEEN 1000 preceding AND CAST(0 AS SMALLINT) following) AS `Sum(f)`
FROM `table` AS `t0`