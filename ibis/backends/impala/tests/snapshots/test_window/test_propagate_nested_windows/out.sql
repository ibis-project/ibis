SELECT
  LAG(`t0`.`f` - LAG(`t0`.`f`) OVER (PARTITION BY `t0`.`g` ORDER BY `t0`.`f` ASC)) OVER (PARTITION BY `t0`.`g` ORDER BY `t0`.`f` ASC) AS `foo`
FROM `alltypes` AS `t0`