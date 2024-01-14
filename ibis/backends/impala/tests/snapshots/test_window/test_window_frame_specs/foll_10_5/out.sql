SELECT
  SUM(`t0`.`d`) OVER (ORDER BY `t0`.`f` ASC NULLS LAST ROWS BETWEEN 10 preceding AND 5 preceding) AS `foo`
FROM `alltypes` AS `t0`