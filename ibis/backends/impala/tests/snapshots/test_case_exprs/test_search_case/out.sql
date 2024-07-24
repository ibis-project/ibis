SELECT
  CASE
    WHEN `t0`.`f` > 0
    THEN `t0`.`d` * 2
    WHEN `t0`.`c` < 0
    THEN `t0`.`a` * 2
    ELSE NULL
  END AS `SearchedCase((Greater(f, 0), Less(c, 0)), (Multiply(d, 2), Multiply(a, 2)), None)`
FROM `alltypes` AS `t0`