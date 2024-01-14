SELECT
  (
    `t0`.`h` OR `t0`.`a` > 0
  ) AND NOT (
    `t0`.`h` AND `t0`.`a` > 0
  ) AS `Xor(h, Greater(a, 0))`
FROM `alltypes` AS `t0`