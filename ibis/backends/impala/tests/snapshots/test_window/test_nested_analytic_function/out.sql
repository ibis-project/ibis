SELECT
  LAG(`t0`.`f` - LAG(`t0`.`f`) OVER (ORDER BY `t0`.`f` ASC)) OVER (ORDER BY `t0`.`f` ASC) AS `foo`
FROM `alltypes` AS `t0`