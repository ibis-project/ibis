SELECT
  LAG(
    `t0`.`f` - LAG(`t0`.`f`) OVER (PARTITION BY `t0`.`g` ORDER BY `t0`.`f` ASC NULLS LAST)
  ) OVER (PARTITION BY `t0`.`g` ORDER BY `t0`.`f` ASC NULLS LAST) AS `foo`
FROM `alltypes` AS `t0`