SELECT
  NOT (
    MAX(`t0`.`f` = 0)
  ) AS `Not(Any(Equals(f, 0)))`
FROM `alltypes` AS `t0`