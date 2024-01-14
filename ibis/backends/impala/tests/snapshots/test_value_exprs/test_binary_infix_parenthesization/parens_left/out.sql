SELECT
  (
    `t0`.`a` + `t0`.`b`
  ) + `t0`.`c` AS `Add(Add(a, b), c)`
FROM `alltypes` AS `t0`