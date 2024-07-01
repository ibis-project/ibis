SELECT
  `t0`.`g` IN ('foo', 'bar', 'baz') AS `InValues(g, ('foo', 'bar', 'baz'))`
FROM `alltypes` AS `t0`