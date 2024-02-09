SELECT
  COUNT(DISTINCT IF(`t0`.`g` = 'A', `t0`.`b`, ARRAY[`t0`.`b`][2])) AS `CountDistinct(b, Equals(g, 'A'))`
FROM `table` AS `t0`