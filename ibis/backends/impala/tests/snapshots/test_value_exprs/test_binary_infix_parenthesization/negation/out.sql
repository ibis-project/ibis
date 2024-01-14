SELECT
  `t0`.`b` + -(
    `t0`.`a` + `t0`.`c`
  ) AS `Add(b, Negate(Add(a, c)))`
FROM `alltypes` AS `t0`