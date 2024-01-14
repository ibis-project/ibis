SELECT
  NULLIF(`t0`.`l_quantity`, `t0`.`l_quantity`) AS `NullIf(l_quantity, l_quantity)`
FROM `tpch_lineitem` AS `t0`