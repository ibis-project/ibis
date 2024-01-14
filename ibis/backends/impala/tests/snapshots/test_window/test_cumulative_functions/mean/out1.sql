SELECT
  AVG(`t0`.`f`) OVER (ORDER BY `t0`.`d` ASC NULLS LAST) AS `foo`
FROM `alltypes` AS `t0`