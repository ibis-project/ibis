SELECT
  NOT (
    `t0`.`g` IN ('foo', 'bar', 'baz')
  ) AS `Not(InValues(g))`
FROM `alltypes` AS `t0`