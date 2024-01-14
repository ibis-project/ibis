SELECT
  CASE `t0`.`g` WHEN 'foo' THEN 'bar' WHEN 'baz' THEN 'qux' ELSE 'default' END AS `SimpleCase(g, 'default')`
FROM `alltypes` AS `t0`