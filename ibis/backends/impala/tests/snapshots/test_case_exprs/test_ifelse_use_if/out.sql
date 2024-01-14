SELECT
  IF(`t0`.`f` > 0, `t0`.`e`, `t0`.`a`) AS `IfElse(Greater(f, 0), e, a)`
FROM `alltypes` AS `t0`