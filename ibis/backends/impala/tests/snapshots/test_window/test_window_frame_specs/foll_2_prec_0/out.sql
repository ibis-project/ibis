SELECT
  SUM(`t0`.`d`) OVER (ORDER BY `t0`.`f` ASC NULLS LAST ROWS BETWEEN CURRENT ROW AND 2 following) AS `foo`
FROM `alltypes` AS `t0`