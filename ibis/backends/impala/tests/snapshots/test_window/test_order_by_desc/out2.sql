SELECT
  LAG(`t0`.`d`) OVER (PARTITION BY `t0`.`g` ORDER BY `t0`.`f` DESC NULLS LAST) AS `foo`,
  MAX(`t0`.`a`) OVER (PARTITION BY `t0`.`g` ORDER BY `t0`.`f` DESC NULLS LAST) AS `Max(a)`
FROM `alltypes` AS `t0`