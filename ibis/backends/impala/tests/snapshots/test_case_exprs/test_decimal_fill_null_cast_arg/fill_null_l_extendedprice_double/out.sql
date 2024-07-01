SELECT
  COALESCE(`t0`.`l_extendedprice`, 0.0) AS `Coalesce((l_extendedprice, 0.0))`
FROM `tpch_lineitem` AS `t0`