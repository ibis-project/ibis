WITH t0 AS (
  SELECT t2.*, t3.*
  FROM tpch_region t2
    INNER JOIN tpch_nation t3
      ON t2.`r_regionkey` = t3.`n_regionkey`
)
SELECT t0.`r_name`, t1.`n_name`
FROM t0
  INNER JOIN t0 t1
    ON t0.`r_regionkey` = t1.`r_regionkey`