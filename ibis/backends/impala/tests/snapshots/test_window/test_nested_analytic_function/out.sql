SELECT
  LAG(`t0`.`f` - LAG(`t0`.`f`) OVER (ORDER BY `t0`.`f` ASC NULLS LAST)) OVER (ORDER BY `t0`.`f` ASC NULLS LAST) AS `foo`
FROM `alltypes` AS `t0`