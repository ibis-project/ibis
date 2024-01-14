SELECT
  SUM(IF(`t0`.`g` IS NULL, 1, 0)) AS `Sum(IfElse(IsNull(g), 1, 0))`
FROM `alltypes` AS `t0`