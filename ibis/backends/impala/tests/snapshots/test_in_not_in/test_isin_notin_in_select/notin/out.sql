SELECT
  *
FROM `alltypes` AS `t0`
WHERE
  NOT (
    `t0`.`g` IN ('foo', 'bar')
  )