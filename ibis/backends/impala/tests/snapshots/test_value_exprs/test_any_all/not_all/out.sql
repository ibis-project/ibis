SELECT
  NOT (
    MIN(`t0`.`f` = 0)
  ) AS `Not(All(Equals(f, 0)))`
FROM `alltypes` AS `t0`