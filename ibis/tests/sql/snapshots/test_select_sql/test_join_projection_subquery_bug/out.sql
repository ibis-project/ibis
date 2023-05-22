WITH t0 AS (
  SELECT t2.`n_nationkey`, t2.`n_name` AS `nation`, t3.`r_name` AS `region`
  FROM tpch_nation t2
    INNER JOIN tpch_region t3
      ON t2.`n_regionkey` = t3.`r_regionkey`
)
SELECT t1.*, t0.*
FROM t0
  INNER JOIN tpch_customer t1
    ON t0.`n_nationkey` = t1.`c_nationkey`