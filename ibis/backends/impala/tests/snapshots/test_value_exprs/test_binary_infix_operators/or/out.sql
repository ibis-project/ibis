SELECT
  `t0`.`h` OR (
    `t0`.`a` > 0
  ) AS `Or(h, Greater(a, 0))`
FROM `alltypes` AS `t0`