SELECT
  SUM(`t0`.`d`) OVER (ORDER BY `t0`.`f` ASC) AS `foo`
FROM `alltypes` AS `t0`