SELECT
  NULLIF(`t0`.`l_quantity` = 0, `t0`.`l_quantity` = 0) AS `NullIf(Equals(l_quantity, 0), Equals(l_quantity, 0))`
FROM `tpch_lineitem` AS `t0`