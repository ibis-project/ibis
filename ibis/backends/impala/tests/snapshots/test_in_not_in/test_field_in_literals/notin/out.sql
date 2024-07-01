SELECT
  NOT (
    `t0`.`g` IN ('foo', 'bar', 'baz')
  ) AS `Not(InValues(g, ('foo', 'bar', 'baz')))`
FROM `alltypes` AS `t0`