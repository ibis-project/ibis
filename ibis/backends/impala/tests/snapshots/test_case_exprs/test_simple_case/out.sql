SELECT
  CASE `t0`.`g` WHEN 'foo' THEN 'bar' WHEN 'baz' THEN 'qux' ELSE 'default' END AS `SimpleCase(g, ('foo', 'baz'), ('bar', 'qux'), 'default')`
FROM `alltypes` AS `t0`