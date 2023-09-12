SELECT *
FROM tpch_nation t0
  INNER JOIN tpch_region t1
    ON t0.`n_regionkey` = t1.`r_regionkey`
  INNER JOIN tpch_customer t2
    ON t0.`n_nationkey` = t2.`c_nationkey`