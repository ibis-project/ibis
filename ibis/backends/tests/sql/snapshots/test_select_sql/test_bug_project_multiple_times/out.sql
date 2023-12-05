WITH t0 AS (
  SELECT t3.*, t4.`n_name`, t5.`r_name`
  FROM tpch_customer t3
    INNER JOIN tpch_nation t4
      ON t3.`c_nationkey` = t4.`n_nationkey`
    INNER JOIN tpch_region t5
      ON t4.`n_regionkey` = t5.`r_regionkey`
),
t1 AS (
  SELECT t0.`n_name`,
         sum(CAST(t0.`c_acctbal` AS double)) AS `Sum(Cast(c_acctbal, float64))`
  FROM t0
  GROUP BY 1
),
t2 AS (
  SELECT t1.*
  FROM t1
  ORDER BY t1.`Sum(Cast(c_acctbal, float64))` DESC
  LIMIT 10
)
SELECT t0.`c_name`, t0.`r_name`, t0.`n_name`
FROM t0
  LEFT SEMI JOIN t2
    ON t0.`n_name` = t2.`n_name`