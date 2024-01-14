SELECT
  (
    `t0`.`a` + `t0`.`b`
  ) IS NULL AS `IsNull(Add(a, b))`
FROM `alltypes` AS `t0`