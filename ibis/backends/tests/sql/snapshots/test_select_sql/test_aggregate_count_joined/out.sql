SELECT count(1) AS `CountStar()`
FROM (
  SELECT t2.*, t1.`r_name` AS `region`
  FROM tpch_region t1
    INNER JOIN tpch_nation t2
      ON t1.`r_regionkey` = t2.`n_regionkey`
) t0